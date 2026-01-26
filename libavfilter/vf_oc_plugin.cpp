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
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#define class clazz
#include "framesync.h"
#undef class
#include "video.h"
}

#include <dlfcn.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "quink_oc_plugin.h"

/**
 * Custom cv::MatAllocator that ties cv::Mat's reference count to AVFrame's.
 *
 * This allocator is used to wrap AVFrame data in cv::Mat without copying.
 * When cv::Mat's reference count reaches zero, the AVFrame is released.
 * This prevents plugins from "stealing" the data through Mat's refcount mechanism.
 *
 * Key behaviors:
 * - allocate(): Returns AVFrame's data pointer, increments AVFrame refcount
 * - deallocate(): Decrements AVFrame refcount (may free the frame)
 * - copy(): Creates a deep copy (normal OpenCV allocation)
 *
 * This ensures that even if a plugin does:
 *   cv::Mat saved = input_mat;  // Just copies header, not data
 * The data will become invalid after process() returns because we'll
 * explicitly release our AVFrame reference.
 */
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
        /* This should not be called for our use case */
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

        /* Release the AVFrame reference */
        if (u->userdata) {
            AVFrame* frame = static_cast<AVFrame*>(u->userdata);
            av_frame_free(&frame);
            u->userdata = nullptr;
        }

        u->data = u->origdata = nullptr;
        delete u;
    }
};

/* Local constants */
#define OC_PLUGIN_MAX_INPUTS  8
#define OC_PLUGIN_MAX_OUTPUTS 8

/**
 * Pixel format mapping table: AVPixelFormat <-> OpenCV type
 */
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

/**
 * Convert AV pixel format to OpenCV type
 */
static int av_to_cv_type(enum AVPixelFormat fmt)
{
    for (int i = 0; pix_fmt_map[i].av_fmt != AV_PIX_FMT_NONE; i++) {
        if (pix_fmt_map[i].av_fmt == fmt)
            return pix_fmt_map[i].cv_type;
    }
    return -1;
}

/**
 * Filter context structure
 */
struct OCPluginFilterContext {
    const AVClass *av_class;

    /* User options */
    char *plugin_path;      ///< Path to plugin shared library
    char *plugin_params;    ///< Parameters passed to plugin
    int nb_inputs;          ///< Number of inputs
    int nb_outputs;         ///< Number of outputs
    int shortest;           ///< Terminate when shortest input ends

    /* Plugin handle and instance */
    void *dl_handle;                             ///< dlopen handle
    const QuinkOCPluginDescriptor *descriptor;   ///< Plugin descriptor
    QuinkOCPlugin *plugin;                       ///< Plugin instance

    /* Multi-input frame sync */
    FFFrameSync fs;
    AVFrame **input_frames;  ///< Temporary storage for input frames

    /* cv::Mat vectors for processing (allocated once, reused per frame) */
    std::vector<cv::Mat> input_mats;
    std::vector<cv::Mat> output_mats;

    /* Output dimensions (may differ from input) */
    int out_width;
    int out_height;
    int out_cv_type;
    enum AVPixelFormat out_pix_fmt;  ///< Output pixel format

    /* Configuration state */
    int configured;

    /* Buffering/flush state */
    int eof_received;           ///< EOF received on input
    int flushing;               ///< Currently flushing buffered frames
    int64_t last_pts;           ///< Last output PTS (for flush frames)
    AVRational time_base;       ///< Output time base
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
    int cv_type = av_to_cv_type(fmt);
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
 * Allocate an output frame
 */
static AVFrame* alloc_output_frame(AVFilterContext *ctx, int64_t pts)
{
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(ctx->priv);
    AVFilterLink *outlink = ctx->outputs[0];

    AVFrame *out = ff_get_video_buffer(outlink, s->out_width, s->out_height);
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
    AVFilterLink *outlink = ctx->outputs[0];
    int ret;

    s->flushing = 1;

    while (1) {
        /* Allocate output frame for flush */
        AVFrame *out = alloc_output_frame(ctx, s->last_pts);
        if (!out)
            return AVERROR(ENOMEM);

        /* Wrap as cv::Mat */
        s->output_mats[0] = avframe_to_mat(out, s->out_pix_fmt, false);
        if (s->output_mats[0].empty()) {
            av_frame_free(&out);
            return AVERROR(EINVAL);
        }

        /* Call plugin flush */
        bool has_frame = s->plugin->flush(s->output_mats);

        s->output_mats[0].release();

        if (!has_frame) {
            av_frame_free(&out);
            break;
        }

        /* Increment PTS for flushed frames */
        s->last_pts++;

        ret = ff_filter_frame(outlink, out);
        if (ret < 0)
            return ret;
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
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *out;

    /* Allocate output frame */
    out = ff_get_video_buffer(outlink, s->out_width, s->out_height);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);

    /* Wrap frames as cv::Mat (zero-copy) */
    /* Input: tie refcount so plugin cannot steal data via Mat's refcount */
    s->input_mats[0] = avframe_to_mat(in, static_cast<AVPixelFormat>(inlink->format), true);
    /* Output: simple view, we own the AVFrame */
    s->output_mats[0] = avframe_to_mat(out, static_cast<AVPixelFormat>(outlink->format), false);

    if (s->input_mats[0].empty() || s->output_mats[0].empty()) {
        av_log(ctx, AV_LOG_ERROR, "Failed to wrap frames as cv::Mat\n");
        av_frame_free(&in);
        av_frame_free(&out);
        return AVERROR(EINVAL);
    }

    QuinkOCProcessResult result = s->plugin->process(s->input_mats, s->output_mats);

    /* Clear Mat references to release AVFrame refcounts */
    s->input_mats[0].release();
    s->output_mats[0].release();

    av_frame_free(&in);

    if (result == QuinkOCProcessResult::ERROR) {
        av_log(ctx, AV_LOG_ERROR, "Plugin processing failed\n");
        av_frame_free(&out);
        return AVERROR_EXTERNAL;
    }

    if (result == QuinkOCProcessResult::TRY_AGAIN) {
        /* Plugin is buffering, don't output frame yet */
        av_frame_free(&out);
        return 0;
    }

    /* Save PTS for flush */
    s->last_pts = out->pts;
    s->time_base = outlink->time_base;

    return ff_filter_frame(outlink, out);
}

/**
 * Process frames from framesync (for multi-input mode)
 */
static int process_frame_multi(FFFrameSync *fs)
{
    AVFilterContext *ctx = fs->parent;
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(fs->opaque);
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame **inputs = s->input_frames;
    AVFrame *out;
    int ret;

    /* Get all input frames */
    for (int i = 0; i < s->nb_inputs; i++) {
        ret = ff_framesync_get_frame(&s->fs, i, &inputs[i], 0);
        if (ret < 0)
            return ret;
    }

    /* Allocate output frame */
    out = ff_get_video_buffer(outlink, s->out_width, s->out_height);
    if (!out)
        return AVERROR(ENOMEM);

    /* Copy properties from first input */
    av_frame_copy_props(out, inputs[0]);
    out->pts = av_rescale_q(s->fs.pts, s->fs.time_base, outlink->time_base);

    /* Wrap all input frames as cv::Mat (zero-copy with tied refcount) */
    for (int i = 0; i < s->nb_inputs; i++) {
        s->input_mats[i] = avframe_to_mat(inputs[i], 
            static_cast<AVPixelFormat>(ctx->inputs[i]->format), true);
        if (s->input_mats[i].empty()) {
            av_log(ctx, AV_LOG_ERROR, "Failed to wrap input %d as cv::Mat\n", i);
            av_frame_free(&out);
            return AVERROR(EINVAL);
        }
    }

    /* Wrap output frame as cv::Mat (simple view, we own the AVFrame) */
    s->output_mats[0] = avframe_to_mat(out, 
        static_cast<AVPixelFormat>(outlink->format), false);
    if (s->output_mats[0].empty()) {
        av_log(ctx, AV_LOG_ERROR, "Failed to wrap output as cv::Mat\n");
        av_frame_free(&out);
        return AVERROR(EINVAL);
    }

    QuinkOCProcessResult result = s->plugin->process(s->input_mats, s->output_mats);

    /* Clear Mat references to release AVFrame refcounts */
    for (int i = 0; i < s->nb_inputs; i++)
        s->input_mats[i].release();
    s->output_mats[0].release();

    if (result == QuinkOCProcessResult::ERROR) {
        av_log(ctx, AV_LOG_ERROR, "Plugin processing failed\n");
        av_frame_free(&out);
        return AVERROR_EXTERNAL;
    }

    if (result == QuinkOCProcessResult::TRY_AGAIN) {
        /* Plugin is buffering, don't output frame yet */
        av_frame_free(&out);
        return 0;
    }

    /* Save PTS for flush */
    s->last_pts = out->pts;
    s->time_base = outlink->time_base;

    return ff_filter_frame(outlink, out);
}

/**
 * Activate callback - handles both multi-input sync and single-input with flush
 */
static int activate(AVFilterContext *ctx)
{
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(ctx->priv);
    AVFilterLink *outlink = ctx->outputs[0];
    int ret;

    if (s->nb_inputs > 1) {
        /* Multi-input mode: use framesync */
        ret = ff_framesync_activate(&s->fs);
        if (ret < 0)
            return ret;

        /* Check for EOF and flush */
        if (ff_outlink_get_status(outlink) && !s->flushing) {
            ret = flush_plugin(ctx);
            if (ret < 0)
                return ret;
        }
        return 0;
    }

    /* Single-input mode */
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
        /* EOF received, flush plugin */
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
    int out_w, out_h;
    int cv_type;

    /* Get OpenCV type for input format */
    cv_type = av_to_cv_type(static_cast<AVPixelFormat>(inlink->format));
    if (cv_type < 0) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported pixel format for OpenCV\n");
        return AVERROR(EINVAL);
    }

    /* Get output dimensions from plugin config callback */
    out_w = inlink->w;
    out_h = inlink->h;

    if (!s->plugin->configure(0, inlink->w, inlink->h, cv_type, out_w, out_h)) {
        av_log(ctx, AV_LOG_ERROR, "Plugin config failed for input 0\n");
        return AVERROR(EINVAL);
    }

    s->out_width = out_w;
    s->out_height = out_h;
    s->out_cv_type = cv_type;
    s->out_pix_fmt = static_cast<AVPixelFormat>(outlink->format);

    outlink->w = out_w;
    outlink->h = out_h;
    outlink->time_base = inlink->time_base;
    outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;
    ol->frame_rate = il->frame_rate;

    s->time_base = outlink->time_base;

    /* For multi-input mode, configure framesync */
    if (s->nb_inputs > 1) {
        /* Configure additional inputs */
        for (int i = 1; i < s->nb_inputs; i++) {
            AVFilterLink *link = ctx->inputs[i];
            int w = out_w, h = out_h;
            int link_cv_type = av_to_cv_type(static_cast<AVPixelFormat>(link->format));
            
            if (!s->plugin->configure(i, link->w, link->h, link_cv_type, w, h)) {
                av_log(ctx, AV_LOG_ERROR, "Plugin config failed for input %d\n", i);
                return AVERROR(EINVAL);
            }
        }

        /* Initialize framesync */
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

    /* Currently only single output is supported */
    if (s->nb_outputs > 1) {
        av_log(ctx, AV_LOG_WARNING, "Multiple outputs not fully supported yet, using 1\n");
        s->nb_outputs = 1;
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

    return 0;
}

/**
 * Uninitialize filter
 */
static av_cold void uninit(AVFilterContext *ctx)
{
    OCPluginFilterContext *s = static_cast<OCPluginFilterContext*>(ctx->priv);

    /* Uninitialize framesync */
    if (s->nb_inputs > 1)
        ff_framesync_uninit(&s->fs);

    av_freep(&s->input_frames);

    /* Clear cv::Mat vectors (releases any remaining references) */
    s->input_mats.clear();
    s->output_mats.clear();

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

static const AVFilterPad oc_plugin_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_output,
    },
};

extern "C"
const FFFilter ff_vf_oc_plugin = {
    .p.name        = "oc_plugin",
    .p.description = NULL_IF_CONFIG_SMALL("Apply processing using external OpenCV plugin."),
    .p.priv_class  = &oc_plugin_class,
    .p.flags       = AVFILTER_FLAG_DYNAMIC_INPUTS,
    .priv_size     = sizeof(OCPluginFilterContext),
    .init          = init,
    .uninit        = uninit,
    .activate      = activate,
    FILTER_OUTPUTS(oc_plugin_outputs),
    FILTER_QUERY_FUNC2(query_formats),
};
