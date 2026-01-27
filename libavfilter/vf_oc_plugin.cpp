/*
 * Copyright (c) 2026 Zhao Zhili <quinkblack@foxmail.com>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

extern "C" {
#include "config.h"

#include "libavutil/avstring.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#define class clazz
#include "framesync.h"
#undef class
#include "video.h"
}

#ifdef _WIN32
#include "compat/w32dlfcn.h"
#else
#include <dlfcn.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "quink_oc_plugin.h"

class AVFrameMatAllocator : public cv::MatAllocator {
public:
    /**
     * Wrap AVFrame data into MatAllocator context.
     * This is called via cv::Mat constructor with custom allocator.
     */
    static cv::UMatData* createUMatData(AVFrame* frame, int cv_type) {
        if (!frame || !frame->data[0])
            return nullptr;

        /* Increment AVFrame reference count */
        AVFrame* ref_frame = av_frame_clone(frame);
        if (!ref_frame)
            return nullptr;

        cv::UMatData* u = new cv::UMatData(getAllocator());
        u->data = u->origdata = ref_frame->data[0];
        u->size = static_cast<size_t>(ref_frame->linesize[0]) * ref_frame->height;
        u->userdata = ref_frame;  /* Store AVFrame pointer for deallocation */
        u->flags |= cv::UMatData::USER_ALLOCATED;  /* We manage the memory */
        return u;
    }

    static AVFrameMatAllocator* getAllocator() {
        static AVFrameMatAllocator instance;
        return &instance;
    }

    cv::UMatData* allocate(int dims, const int* sizes, int type,
                           void* data, size_t* step, cv::AccessFlag flags,
                           cv::UMatUsageFlags usageFlags) const override {
        (void)dims; (void)sizes; (void)type; (void)data;
        (void)step; (void)flags; (void)usageFlags;
        return nullptr;
    }

    bool allocate(cv::UMatData* u, cv::AccessFlag accessFlags,
                  cv::UMatUsageFlags usageFlags) const override {
        (void)u; (void)accessFlags; (void)usageFlags;
        return false;
    }

    void deallocate(cv::UMatData* u) const override {
        if (!u)
            return;

        if (u->userdata) {
            AVFrame* frame = static_cast<AVFrame*>(u->userdata);
            av_frame_free(&frame);
            u->userdata = nullptr;
        }

        u->data = u->origdata = nullptr;
        delete u;
    }
};

#define OC_PLUGIN_MAX_INPUTS  8
#define OC_PLUGIN_MAX_OUTPUTS 8

static const struct {
    enum AVPixelFormat av_fmt;
    int cv_type;
} pix_fmt_map[] = {
    { AV_PIX_FMT_BGR24,     CV_8UC3  },
    { AV_PIX_FMT_BGRA,      CV_8UC4  },
    { AV_PIX_FMT_GRAY8,     CV_8UC1  },
    { AV_PIX_FMT_GRAY16LE,  CV_16UC1 },
    { AV_PIX_FMT_BGR48LE,   CV_16UC3 },
    { AV_PIX_FMT_NONE,      -1       },
};

static int pixfmt_to_cv_type(enum AVPixelFormat fmt)
{
    for (int i = 0; pix_fmt_map[i].av_fmt != AV_PIX_FMT_NONE; i++) {
        if (pix_fmt_map[i].av_fmt == fmt)
            return pix_fmt_map[i].cv_type;
    }
    return -1;
}

struct OCPluginFilterContext {
    const AVClass *clazz;

    /* User options */
    char *plugin_path;
    char *plugin_params;
    int nb_inputs;
    int nb_outputs;
    int shortest;

    /* Plugin handle and instance */
    void *dl_handle;
    const QuinkOCPluginDescriptor *descriptor;
    QuinkOCPlugin *plugin;

    /* Multi-input frame sync */
    FFFrameSync fs;
    AVFrame **input_frames;

    /* cv::Mat vectors for processing (allocated once, reused per frame) */
    std::vector<cv::Mat> input_mats;
    std::vector<cv::Mat> output_mats;

    std::vector<QuinkOCFrameConfig> out_configs;
    std::vector<AVPixelFormat> out_pix_fmts;

    int configured;
    int use_framesync;

    /* Buffering/flush state */
    int eof_received;
    int flushing;
    int64_t last_pts;
    AVRational time_base;
};

/**
 * Load plugin from shared library
 */
static int load_plugin(AVFilterContext *ctx)
{
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(ctx->priv);
    QuinkOCPluginGetDescriptorFunc get_descriptor;
    const char *error;

    if (!s->plugin_path || !s->plugin_path[0]) {
        av_log(ctx, AV_LOG_ERROR, "No plugin path specified\n");
        return AVERROR(EINVAL);
    }

    /* Open shared library */
    s->dl_handle = dlopen(s->plugin_path, RTLD_NOW | RTLD_LOCAL);
    if (!s->dl_handle) {
        error = dlerror();
        av_log(ctx, AV_LOG_ERROR, "Failed to load plugin '%s': %s\n",
               s->plugin_path, error ? error : "unknown error");
        return AVERROR(EINVAL);
    }

    /* Clear any existing error */
    dlerror();

    /* Load the single entry point function */
    get_descriptor = reinterpret_cast<QuinkOCPluginGetDescriptorFunc>(
        dlsym(s->dl_handle, QUINK_OC_PLUGIN_DESCRIPTOR_SYMBOL));
    if (!get_descriptor) {
        error = dlerror();
        av_log(ctx, AV_LOG_ERROR, "Plugin missing '%s' symbol: %s\n",
               QUINK_OC_PLUGIN_DESCRIPTOR_SYMBOL, error ? error : "unknown error");
        return AVERROR(EINVAL);
    }

    /* Get the static descriptor */
    s->descriptor = get_descriptor();
    if (!s->descriptor) {
        av_log(ctx, AV_LOG_ERROR, "Plugin returned NULL descriptor\n");
        return AVERROR(EINVAL);
    }

    /* Validate descriptor */
    if (s->descriptor->api_version != QUINK_OC_PLUGIN_API_VERSION) {
        av_log(ctx, AV_LOG_ERROR, "Plugin API version mismatch: expected %d, got %d\n",
               QUINK_OC_PLUGIN_API_VERSION, s->descriptor->api_version);
        return AVERROR(EINVAL);
    }

    if (!s->descriptor->create || !s->descriptor->destroy) {
        av_log(ctx, AV_LOG_ERROR, "Plugin descriptor missing create/destroy functions\n");
        return AVERROR(EINVAL);
    }

    return 0;
}

/**
 * Initialize plugin with configuration
 */
static int init_plugin(AVFilterContext *ctx)
{
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(ctx->priv);

    /* Create plugin instance */
    s->plugin = s->descriptor->create();
    if (!s->plugin) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create plugin instance\n");
        return AVERROR(ENOMEM);
    }

    /* Validate input/output counts against plugin requirements */
    int plugin_inputs = s->plugin->numInputs();
    int plugin_outputs = s->plugin->numOutputs();

    if (s->nb_inputs != plugin_inputs) {
        av_log(ctx, AV_LOG_WARNING, "Plugin expects %d inputs, filter configured with %d\n",
               plugin_inputs, s->nb_inputs);
    }
    if (s->nb_outputs != plugin_outputs) {
        av_log(ctx, AV_LOG_WARNING, "Plugin expects %d outputs, filter configured with %d\n",
               plugin_outputs, s->nb_outputs);
    }

    /* Initialize plugin */
    if (!s->plugin->init(s->plugin_params)) {
        av_log(ctx, AV_LOG_ERROR, "Plugin initialization failed\n");
        s->descriptor->destroy(s->plugin);
        s->plugin = nullptr;
        return AVERROR(EINVAL);
    }

    av_log(ctx, AV_LOG_INFO, "Loaded plugin: %s - %s\n",
           s->descriptor->name, s->descriptor->description);

    /* Reserve cv::Mat vectors (actual Mats created per-frame) */
    s->input_mats.resize(s->nb_inputs);
    s->output_mats.resize(s->nb_outputs);

    return 0;
}

/**
 * Query supported formats
 */
static int query_formats(const AVFilterContext *ctx,
                         AVFilterFormatsConfig **cfg_in,
                         AVFilterFormatsConfig **cfg_out)
{
    /* Support common formats that can be converted to cv::Mat */
    static const int default_fmts[] = {
        AV_PIX_FMT_BGR24,
        AV_PIX_FMT_BGRA,
        AV_PIX_FMT_GRAY8,
        AV_PIX_FMT_GRAY16LE,
        AV_PIX_FMT_BGR48LE,
        AV_PIX_FMT_NONE
    };

    AVFilterFormats *formats = ff_make_format_list(default_fmts);
    if (!formats)
        return AVERROR(ENOMEM);

    return ff_set_common_formats2(ctx, cfg_in, cfg_out, formats);
}

/**
 * Wrap AVFrame as cv::Mat with tied reference counting (zero-copy)
 *
 * This function creates a cv::Mat that directly references the AVFrame's
 * pixel data without any memory copy. The key innovation is using a custom
 * cv::MatAllocator to tie the Mat's lifecycle to the AVFrame's refcount.
 *
 * Behavior:
 * - The cv::Mat shares ownership of the underlying AVFrame data
 * - If the plugin tries to keep the Mat (e.g., `saved = input_mat`), it will
 *   hold a reference to the AVFrame until the Mat is destroyed
 * - When all Mat references are released, the AVFrame refcount decrements
 *
 * @param frame The AVFrame to wrap (must have continuous data in data[0])
 * @param fmt   The pixel format of the frame
 * @param tie_refcount If true, tie Mat's refcount to AVFrame's refcount.
 *                     If false, create a simple view (Mat won't own data).
 * @return cv::Mat wrapping the frame data, or empty Mat on unsupported format
 */
static cv::Mat avframe_to_mat(AVFrame *frame, enum AVPixelFormat fmt, bool tie_refcount = true)
{
    int cv_type = pixfmt_to_cv_type(fmt);
    if (cv_type < 0 || !frame || !frame->data[0])
        return cv::Mat();

    if (tie_refcount) {
        /*
         * Create Mat with custom allocator that ties to AVFrame refcount.
         * This ensures proper lifecycle management even if the plugin
         * attempts to retain the Mat beyond process().
         */
        cv::UMatData* u = AVFrameMatAllocator::createUMatData(frame, cv_type);
        if (!u)
            return cv::Mat();

        cv::Mat mat(frame->height, frame->width, cv_type,
                    frame->data[0], static_cast<size_t>(frame->linesize[0]));
        mat.u = u;
        mat.addref();  /* Initial reference */
        return mat;
    } else {
        /*
         * Simple view without ownership (for output frames where we
         * already own the AVFrame and will manage its lifecycle).
         */
        return cv::Mat(frame->height, frame->width, cv_type,
                       frame->data[0], static_cast<size_t>(frame->linesize[0]));
    }
}

/**
 * Allocate an output frame for specified output index
 */
static AVFrame* alloc_output_frame(AVFilterContext *ctx, int output_idx, int64_t pts)
{
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(ctx->priv);
    AVFilterLink *outlink = ctx->outputs[output_idx];

    AVFrame *out = ff_get_video_buffer(outlink, s->out_configs[output_idx].width, s->out_configs[output_idx].height);
    if (!out)
        return nullptr;

    out->pts = pts;
    return out;
}

/**
 * Flush buffered frames from plugin
 */
static int flush_plugin(AVFilterContext *ctx)
{
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(ctx->priv);
    int ret;

    s->flushing = 1;

    std::vector<AVFrame*> out_frames(s->nb_outputs);

    while (1) {
        /* Allocate output frames for flush */
        for (int i = 0; i < s->nb_outputs; i++) {
            out_frames[i] = alloc_output_frame(ctx, i, s->last_pts);
            if (!out_frames[i]) {
                for (int j = 0; j < i; j++)
                    av_frame_free(&out_frames[j]);
                return AVERROR(ENOMEM);
            }
            s->output_mats[i] = avframe_to_mat(out_frames[i], s->out_pix_fmts[i], false);
            if (s->output_mats[i].empty()) {
                for (int j = 0; j <= i; j++)
                    av_frame_free(&out_frames[j]);
                return AVERROR(EINVAL);
            }
        }

        bool has_frame = s->plugin->flush(s->output_mats);

        for (int i = 0; i < s->nb_outputs; i++)
            s->output_mats[i].release();

        if (!has_frame) {
            for (int i = 0; i < s->nb_outputs; i++)
                av_frame_free(&out_frames[i]);
            break;
        }

        s->last_pts++;

        for (int i = 0; i < s->nb_outputs; i++) {
            ret = ff_filter_frame(ctx->outputs[i], out_frames[i]);
            if (ret < 0)
                return ret;
        }
    }

    s->flushing = 0;
    return 0;
}

/**
 * Process a single frame (for single-input mode)
 */
static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(ctx->priv);
    std::vector<AVFrame*> out_frames(s->nb_outputs);
    int ret;

    /* Allocate output frames */
    for (int i = 0; i < s->nb_outputs; i++) {
        AVFilterLink *outlink = ctx->outputs[i];
        out_frames[i] = ff_get_video_buffer(outlink, s->out_configs[i].width, s->out_configs[i].height);
        if (!out_frames[i]) {
            for (int j = 0; j < i; j++)
                av_frame_free(&out_frames[j]);
            av_frame_free(&in);
            return AVERROR(ENOMEM);
        }
        av_frame_copy_props(out_frames[i], in);
    }

    /* Wrap input frame as cv::Mat (tie refcount) */
    s->input_mats[0] = avframe_to_mat(in, static_cast<AVPixelFormat>(inlink->format), true);
    if (s->input_mats[0].empty()) {
        av_log(ctx, AV_LOG_ERROR, "Failed to wrap input frame as cv::Mat\n");
        for (int i = 0; i < s->nb_outputs; i++)
            av_frame_free(&out_frames[i]);
        av_frame_free(&in);
        return AVERROR(EINVAL);
    }

    /* Wrap output frames as cv::Mat (simple view) */
    for (int i = 0; i < s->nb_outputs; i++) {
        s->output_mats[i] = avframe_to_mat(out_frames[i], s->out_pix_fmts[i], false);
        if (s->output_mats[i].empty()) {
            av_log(ctx, AV_LOG_ERROR, "Failed to wrap output %d as cv::Mat\n", i);
            s->input_mats[0].release();
            for (int j = 0; j < s->nb_outputs; j++)
                av_frame_free(&out_frames[j]);
            av_frame_free(&in);
            return AVERROR(EINVAL);
        }
    }

    QuinkOCProcessResult result = s->plugin->process(s->input_mats, s->output_mats);

    /* Clear Mat references */
    s->input_mats[0].release();
    for (int i = 0; i < s->nb_outputs; i++)
        s->output_mats[i].release();

    av_frame_free(&in);

    if (result == QuinkOCProcessResult::ERROR) {
        av_log(ctx, AV_LOG_ERROR, "Plugin processing failed\n");
        for (int i = 0; i < s->nb_outputs; i++)
            av_frame_free(&out_frames[i]);
        return AVERROR_EXTERNAL;
    }

    if (result == QuinkOCProcessResult::TRY_AGAIN) {
        for (int i = 0; i < s->nb_outputs; i++)
            av_frame_free(&out_frames[i]);
        return 0;
    }

    /* Save PTS for flush */
    s->last_pts = out_frames[0]->pts;
    s->time_base = ctx->outputs[0]->time_base;

    /* Output all frames */
    for (int i = 0; i < s->nb_outputs; i++) {
        ret = ff_filter_frame(ctx->outputs[i], out_frames[i]);
        if (ret < 0)
            return ret;
    }

    return 0;
}

/**
 * Process frames from framesync (for multi-input mode)
 */
static int process_frame_multi(FFFrameSync *fs)
{
    AVFilterContext *ctx = fs->parent;
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(fs->opaque);
    AVFrame **inputs = s->input_frames;
    std::vector<AVFrame*> out_frames(s->nb_outputs);
    int ret;

    /* Get all input frames */
    for (int i = 0; i < s->nb_inputs; i++) {
        ret = ff_framesync_get_frame(&s->fs, i, &inputs[i], 0);
        if (ret < 0)
            return ret;
    }

    /* Allocate output frames */
    for (int i = 0; i < s->nb_outputs; i++) {
        AVFilterLink *outlink = ctx->outputs[i];
        out_frames[i] = ff_get_video_buffer(outlink, s->out_configs[i].width, s->out_configs[i].height);
        if (!out_frames[i]) {
            for (int j = 0; j < i; j++)
                av_frame_free(&out_frames[j]);
            return AVERROR(ENOMEM);
        }
        av_frame_copy_props(out_frames[i], inputs[0]);
        out_frames[i]->pts = av_rescale_q(s->fs.pts, s->fs.time_base, outlink->time_base);
    }

    /* Wrap all input frames as cv::Mat (zero-copy with tied refcount) */
    for (int i = 0; i < s->nb_inputs; i++) {
        s->input_mats[i] = avframe_to_mat(inputs[i], 
            static_cast<AVPixelFormat>(ctx->inputs[i]->format), true);
        if (s->input_mats[i].empty()) {
            av_log(ctx, AV_LOG_ERROR, "Failed to wrap input %d as cv::Mat\n", i);
            for (int j = 0; j < s->nb_outputs; j++)
                av_frame_free(&out_frames[j]);
            return AVERROR(EINVAL);
        }
    }

    /* Wrap output frames as cv::Mat (simple view) */
    for (int i = 0; i < s->nb_outputs; i++) {
        s->output_mats[i] = avframe_to_mat(out_frames[i], s->out_pix_fmts[i], false);
        if (s->output_mats[i].empty()) {
            av_log(ctx, AV_LOG_ERROR, "Failed to wrap output %d as cv::Mat\n", i);
            for (int j = 0; j < s->nb_inputs; j++)
                s->input_mats[j].release();
            for (int j = 0; j < s->nb_outputs; j++)
                av_frame_free(&out_frames[j]);
            return AVERROR(EINVAL);
        }
    }

    QuinkOCProcessResult result = s->plugin->process(s->input_mats, s->output_mats);

    /* Clear Mat references */
    for (int i = 0; i < s->nb_inputs; i++)
        s->input_mats[i].release();
    for (int i = 0; i < s->nb_outputs; i++)
        s->output_mats[i].release();

    if (result == QuinkOCProcessResult::ERROR) {
        av_log(ctx, AV_LOG_ERROR, "Plugin processing failed\n");
        for (int i = 0; i < s->nb_outputs; i++)
            av_frame_free(&out_frames[i]);
        return AVERROR_EXTERNAL;
    }

    if (result == QuinkOCProcessResult::TRY_AGAIN) {
        for (int i = 0; i < s->nb_outputs; i++)
            av_frame_free(&out_frames[i]);
        return 0;
    }

    /* Save PTS for flush */
    s->last_pts = out_frames[0]->pts;
    s->time_base = ctx->outputs[0]->time_base;

    /* Output all frames */
    for (int i = 0; i < s->nb_outputs; i++) {
        ret = ff_filter_frame(ctx->outputs[i], out_frames[i]);
        if (ret < 0)
            return ret;
    }

    return 0;
}

static int activate(AVFilterContext *ctx)
{
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(ctx->priv);
    AVFilterLink *outlink = ctx->outputs[0];
    int ret;

    if (s->use_framesync) {
        /* Multi-input + single-output mode (N:1): use framesync */
        ret = ff_framesync_activate(&s->fs);
        if (ret < 0)
            return ret;

        if (ff_outlink_get_status(outlink) && !s->flushing) {
            ret = flush_plugin(ctx);
            if (ret < 0)
                return ret;
        }
        return 0;
    }

    /* Single-input mode (1:1 or 1:N) */
    AVFilterLink *inlink = ctx->inputs[0];
    AVFrame *frame = NULL;
    int status;
    int64_t pts;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    ret = ff_inlink_consume_frame(inlink, &frame);
    if (ret < 0)
        return ret;

    if (frame) {
        ret = filter_frame(inlink, frame);
        if (ret < 0)
            return ret;
    }

    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF && !s->flushing) {
            ret = flush_plugin(ctx);
            if (ret < 0)
                return ret;
        }
        ff_outlink_set_status(outlink, status, pts);
        return 0;
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return 0;
}

/**
 * Configure output link
 */
static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(ctx->priv);
    AVFilterLink *inlink = ctx->inputs[0];
    FilterLink *il = ff_filter_link(inlink);
    FilterLink *ol = ff_filter_link(outlink);
    FFFrameSyncIn *in;
    int ret;

    /* Collect all input configurations */
    std::vector<QuinkOCFrameConfig> input_configs(s->nb_inputs);
    for (int i = 0; i < s->nb_inputs; i++) {
        AVFilterLink *link = ctx->inputs[i];
        int link_cv_type = pixfmt_to_cv_type(static_cast<AVPixelFormat>(link->format));
        if (link_cv_type < 0) {
            av_log(ctx, AV_LOG_ERROR, "Unsupported pixel format for input %d\n", i);
            return AVERROR(EINVAL);
        }
        input_configs[i].width = link->w;
        input_configs[i].height = link->h;
        input_configs[i].cv_type = link_cv_type;
    }

    /* Initialize output configurations: each output defaults to corresponding input */
    std::vector<QuinkOCFrameConfig> output_configs(s->nb_outputs);
    for (int i = 0; i < s->nb_outputs; i++) {
        int src_idx = (i < s->nb_inputs) ? i : 0;
        output_configs[i].width = input_configs[src_idx].width;
        output_configs[i].height = input_configs[src_idx].height;
        output_configs[i].cv_type = 0;
    }

    if (!s->plugin->configure(input_configs, output_configs)) {
        av_log(ctx, AV_LOG_ERROR, "Plugin configure failed\n");
        return AVERROR(EINVAL);
    }

    /* Store output configurations */
    s->out_configs = output_configs;
    s->out_pix_fmts.resize(s->nb_outputs);
    for (int i = 0; i < s->nb_outputs; i++) {
        s->out_pix_fmts[i] = static_cast<AVPixelFormat>(ctx->outputs[i]->format);
    }

    outlink->w = s->out_configs[0].width;
    outlink->h = s->out_configs[0].height;
    outlink->time_base = inlink->time_base;
    outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;
    ol->frame_rate = il->frame_rate;

    /* Configure additional outputs */
    for (int i = 1; i < s->nb_outputs; i++) {
        AVFilterLink *out_i = ctx->outputs[i];
        FilterLink *ol_i = ff_filter_link(out_i);
        out_i->w = s->out_configs[i].width;
        out_i->h = s->out_configs[i].height;
        out_i->time_base = inlink->time_base;
        out_i->sample_aspect_ratio = inlink->sample_aspect_ratio;
        ol_i->frame_rate = il->frame_rate;
    }

    s->time_base = outlink->time_base;

    /* For multi-input + single-output mode, use framesync */
    if (s->nb_inputs > 1 && s->nb_outputs == 1) {
        ret = ff_framesync_init(&s->fs, ctx, s->nb_inputs);
        if (ret < 0)
            return ret;

        s->fs.opaque = s;
        s->fs.on_event = process_frame_multi;

        in = s->fs.in;
        for (int i = 0; i < s->nb_inputs; i++) {
            in[i].time_base = ctx->inputs[i]->time_base;
            in[i].sync = 1;
            in[i].before = EXT_STOP;
            in[i].after = s->shortest ? EXT_STOP : EXT_INFINITY;
        }

        ret = ff_framesync_configure(&s->fs);
        if (ret < 0)
            return ret;

        outlink->time_base = s->fs.time_base;
        s->use_framesync = 1;
    }

    s->configured = 1;
    return 0;
}

/**
 * Initialize filter
 */
static av_cold int init(AVFilterContext *ctx)
{
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(ctx->priv);
    int ret;

    /* Validate input/output counts */
    if (s->nb_inputs < 1 || s->nb_inputs > OC_PLUGIN_MAX_INPUTS) {
        av_log(ctx, AV_LOG_ERROR, "Invalid number of inputs: %d (must be 1-%d)\n",
               s->nb_inputs, OC_PLUGIN_MAX_INPUTS);
        return AVERROR(EINVAL);
    }
    if (s->nb_outputs < 1 || s->nb_outputs > OC_PLUGIN_MAX_OUTPUTS) {
        av_log(ctx, AV_LOG_ERROR, "Invalid number of outputs: %d (must be 1-%d)\n",
               s->nb_outputs, OC_PLUGIN_MAX_OUTPUTS);
        return AVERROR(EINVAL);
    }

    /* Multi-input + multi-output is not supported */
    if (s->nb_inputs > 1 && s->nb_outputs > 1) {
        av_log(ctx, AV_LOG_ERROR,
               "Multi-input + multi-output (N:M) not supported. "
               "Use 1:N or N:1 mode, or chain multiple filters.\n");
        return AVERROR(EINVAL);
    }

    /* Load plugin */
    ret = load_plugin(ctx);
    if (ret < 0)
        return ret;

    /* Initialize plugin */
    ret = init_plugin(ctx);
    if (ret < 0)
        return ret;

    /* Create input pads */
    for (int i = 0; i < s->nb_inputs; i++) {
        AVFilterPad pad = {0};
        pad.type = AVMEDIA_TYPE_VIDEO;

        if (s->nb_inputs == 1) {
            pad.name = av_strdup("default");
            pad.filter_frame = filter_frame;
        } else {
            pad.name = av_asprintf("input%d", i);
        }
        if (!pad.name)
            return AVERROR(ENOMEM);

        ret = ff_append_inpad_free_name(ctx, &pad);
        if (ret < 0)
            return ret;
    }

    /* Allocate frame storage for multi-input mode */
    if (s->nb_inputs > 1) {
        s->input_frames = static_cast<AVFrame**>(av_calloc(s->nb_inputs, sizeof(*s->input_frames)));
        if (!s->input_frames)
            return AVERROR(ENOMEM);
    }

    /* Create output pads */
    for (int i = 0; i < s->nb_outputs; i++) {
        AVFilterPad pad = {0};
        pad.type = AVMEDIA_TYPE_VIDEO;
        pad.config_props = config_output;

        if (s->nb_outputs == 1) {
            pad.name = av_strdup("default");
        } else {
            pad.name = av_asprintf("output%d", i);
        }
        if (!pad.name)
            return AVERROR(ENOMEM);

        ret = ff_append_outpad_free_name(ctx, &pad);
        if (ret < 0)
            return ret;
    }

    return 0;
}

/**
 * Uninitialize filter
 */
static av_cold void uninit(AVFilterContext *ctx)
{
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(ctx->priv);

    /* Uninitialize framesync */
    if (s->use_framesync)
        ff_framesync_uninit(&s->fs);

    av_freep(&s->input_frames);

    /* Clear cv::Mat vectors (releases any remaining references) */
    s->input_mats.clear();
    s->output_mats.clear();
    s->out_configs.clear();
    s->out_pix_fmts.clear();

    /* Uninitialize and destroy plugin */
    if (s->plugin) {
        s->plugin->uninit();
        if (s->descriptor && s->descriptor->destroy)
            s->descriptor->destroy(s->plugin);
        s->plugin = nullptr;
    }

    if (s->dl_handle) {
        dlclose(s->dl_handle);
        s->dl_handle = nullptr;
    }
}

#define OFFSET(x) offsetof(OCPluginFilterContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption oc_plugin_options[] = {
    { "plugin", "Path to OpenCV plugin shared library (.so/.dylib/.dll)",
        OFFSET(plugin_path), AV_OPT_TYPE_STRING, {.str = nullptr}, 0, 0, FLAGS },
    { "p", "Path to OpenCV plugin shared library (.so/.dylib/.dll)",
        OFFSET(plugin_path), AV_OPT_TYPE_STRING, {.str = nullptr}, 0, 0, FLAGS },
    { "params", "Parameters to pass to the plugin",
        OFFSET(plugin_params), AV_OPT_TYPE_STRING, {.str = nullptr}, 0, 0, FLAGS },
    { "inputs", "Number of inputs",
        OFFSET(nb_inputs), AV_OPT_TYPE_INT, {.i64 = 1}, 1, OC_PLUGIN_MAX_INPUTS, FLAGS },
    { "outputs", "Number of outputs",
        OFFSET(nb_outputs), AV_OPT_TYPE_INT, {.i64 = 1}, 1, OC_PLUGIN_MAX_OUTPUTS, FLAGS },
    { "shortest", "Force termination when the shortest input terminates",
        OFFSET(shortest), AV_OPT_TYPE_BOOL, {.i64 = 0}, 0, 1, FLAGS },
    { nullptr }
};

AVFILTER_DEFINE_CLASS(oc_plugin);

extern "C"
const FFFilter ff_vf_oc_plugin = []() {
    FFFilter f = {0};
    f.p.name = "oc_plugin";
    f.p.description = NULL_IF_CONFIG_SMALL("Apply processing using external OpenCV plugin.");
    f.p.inputs = nullptr, f.p.outputs = nullptr,
    f.p.priv_class = &oc_plugin_class,
    f.p.flags = AVFILTER_FLAG_DYNAMIC_INPUTS | AVFILTER_FLAG_DYNAMIC_OUTPUTS,

    f.nb_inputs = 0;
    f.nb_outputs = 0;
    f.formats_state = FF_FILTER_FORMATS_QUERY_FUNC2;
    f.preinit = nullptr;
    f.init = init;
    f.uninit = uninit;
    f.formats.query_func2 = query_formats;
    f.priv_size = sizeof(OCPluginFilterContext);
    f.flags_internal = 0;
    f.process_command = nullptr;
    f.activate = activate;

    return f;
}();
