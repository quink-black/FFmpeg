#include "config_components.h"

#include <stdbool.h>
#include <multimedia/player_framework/native_avcodec_videodecoder.h>

#include "libavutil/fifo.h"
#include "libavutil/hwcontext_oh.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/thread.h"

#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"
#include "hwconfig.h"

typedef struct OHBufferQueueItem {
    uint32_t index;
    OH_AVBuffer *buffer;
} OHBufferQueueItem;

typedef struct OHCodecDecContext {
    AVClass *avclass;
    OH_AVCodec *dec;
    AVBufferRef *dec_ref;

    AVMutex input_mutex;
    AVCond input_cond;
    AVFifo *input_queue;

    AVMutex output_mutex;
    AVCond output_cond;
    AVFifo *output_queue;

    AVPacket pkt;

    int decode_status;
    bool eof_sent;

    bool output_to_window;
    int width;
    int height;
    int stride;
    int slice_height;
    OH_AVPixelFormat pix_fmt;
} OHCodecDecContext;

typedef struct OHCodecBuffer {
    AVBufferRef *dec_ref;
    uint32_t index;
    OH_AVBuffer *buffer;
} OHCodecBuffer;

static const struct {
    OH_AVPixelFormat oh_pix;
    enum AVPixelFormat pix;
} oh_pix_map[] = {
    {AV_PIXEL_FORMAT_NV12,           AV_PIX_FMT_NV12},
    {AV_PIXEL_FORMAT_NV21,           AV_PIX_FMT_NV21},
    {AV_PIXEL_FORMAT_YUVI420,        AV_PIX_FMT_YUV420P},
    {AV_PIXEL_FORMAT_SURFACE_FORMAT, AV_PIX_FMT_OPENHARMONY},
};

static enum AVPixelFormat oh_pix_to_av_pix(OH_AVPixelFormat oh_pix)
{
    for (size_t i = 0; i < FF_ARRAY_ELEMS(oh_pix_map); i++)
        if (oh_pix_map[i].oh_pix == oh_pix)
            return oh_pix_map[i].pix;

    return AV_PIX_FMT_NONE;
}

static int32_t oh_pix_from_av_pix(enum AVPixelFormat pix)
{
    for (size_t i = 0; i < FF_ARRAY_ELEMS(oh_pix_map); i++)
        if (oh_pix_map[i].pix == pix)
            return oh_pix_map[i].oh_pix;

    return 0;
}

static int oh_err_to_averr(OH_AVErrCode err)
{
    switch (err) {
    case AV_ERR_OK:
        return 0;
    case AV_ERR_NO_MEMORY:
        return AVERROR(ENOMEM);
    case AV_ERR_OPERATE_NOT_PERMIT:
        return AVERROR(EPERM);
    case AV_ERR_INVALID_VAL:
        return AVERROR(EINVAL);
    case AV_ERR_IO:
        return AVERROR(EIO);
    case AV_ERR_TIMEOUT:
        return AVERROR(ETIMEDOUT);
    case AV_ERR_UNKNOWN:
        return AVERROR_UNKNOWN;
    case AV_ERR_SERVICE_DIED:
        return AVERROR_EXTERNAL;
    case AV_ERR_INVALID_STATE:
        return AVERROR(EINVAL);
    case AV_ERR_UNSUPPORT:
        return AVERROR(ENOTSUP);
    default:
        return AVERROR_EXTERNAL;
    }
}

static void oh_decode_release(void *opaque, uint8_t *data)
{
    OH_AVCodec *dec = (OH_AVCodec *)data;
    OH_VideoDecoder_Destroy(dec);
}

static int oh_decode_create(OHCodecDecContext *s, AVCodecContext *avctx)
{
    const char *mime;
    switch (avctx->codec_id) {
        case AV_CODEC_ID_H264:
            mime = OH_AVCODEC_MIMETYPE_VIDEO_AVC;
            break;
        case AV_CODEC_ID_HEVC:
            mime = OH_AVCODEC_MIMETYPE_VIDEO_HEVC;
            break;
        default:
            return AVERROR(EINVAL);
    }

    s->dec = OH_VideoDecoder_CreateByMime(mime);
    if (!s->dec) {
        av_log(avctx, AV_LOG_ERROR, "Create decoder by mime %s failed\n", mime);
        return AVERROR_EXTERNAL;
    }
    s->dec_ref = av_buffer_create((uint8_t *)s->dec, 0, oh_decode_release,
                                  NULL, AV_BUFFER_FLAG_READONLY);
    if (!s->dec_ref)
        return AVERROR(ENOMEM);

    return 0;
}

static int oh_decode_set_format(OHCodecDecContext *s, AVCodecContext *avctx)
{
    int ret;
    OHNativeWindow *window = NULL;

    if (avctx->hw_device_ctx) {
        AVHWDeviceContext *device_ctx = (AVHWDeviceContext*)(avctx->hw_device_ctx->data);
        if (device_ctx->type == AV_HWDEVICE_TYPE_OPENHARMONY) {
            AVOHCodecDeviceContext *dev = device_ctx->hwctx;
            window = dev->native_window;
            s->output_to_window = true;
        } else {
            av_log(avctx, AV_LOG_WARNING, "Ignore invalid hw device type %s\n",
                   av_hwdevice_get_type_name(device_ctx->type));
        }
    }

    OH_AVFormat *format = OH_AVFormat_Create();
    if (!format)
        return AVERROR(ENOMEM);

    OH_AVFormat_SetIntValue(format, OH_MD_KEY_WIDTH, avctx->width);
    OH_AVFormat_SetIntValue(format, OH_MD_KEY_HEIGHT, avctx->height);
    if (!s->output_to_window)
        OH_AVFormat_SetIntValue(format, OH_MD_KEY_PIXEL_FORMAT,
                                AV_PIXEL_FORMAT_NV12);
    else
        OH_AVFormat_SetIntValue(format, OH_MD_KEY_PIXEL_FORMAT,
                                AV_PIXEL_FORMAT_SURFACE_FORMAT);
    if (avctx->framerate.num > 0 && avctx->framerate.den > 0)
        OH_AVFormat_SetDoubleValue(format, OH_MD_KEY_FRAME_RATE,
                                   av_q2d(avctx->framerate));
    OH_AVErrCode err = OH_VideoDecoder_Configure(s->dec, format);
    OH_AVFormat_Destroy(format);
    if (err != AV_ERR_OK) {
        ret = oh_err_to_averr(err);
        av_log(avctx, AV_LOG_ERROR, "decoder configure failed, %d, %s\n",
               err, av_err2str(ret));
        return ret;
    }

    if (s->output_to_window) {
        err = OH_VideoDecoder_SetSurface(s->dec, window);
        if (err != AV_ERR_OK) {
            ret = oh_err_to_averr(err);
            av_log(avctx, AV_LOG_ERROR, "set surface failed, %d, %s\n",
                   err, av_err2str(ret));
            return ret;
        }
    }

    s->width = avctx->width;
    s->height = avctx->height;
    s->slice_height = s->height;

    return 0;
}

static void oh_decode_on_err(OH_AVCodec *codec, int32_t err, void *userdata)
{
    AVCodecContext *avctx = userdata;
    OHCodecDecContext *s = avctx->priv_data;

    ff_mutex_lock(&s->input_mutex);
    ff_mutex_lock(&s->output_mutex);
    s->decode_status = oh_err_to_averr(err);
    ff_mutex_unlock(&s->output_mutex);
    ff_mutex_unlock(&s->input_mutex);

    ff_cond_signal(&s->output_cond);
    ff_cond_signal(&s->input_cond);
}

static void oh_decode_on_stream_changed(OH_AVCodec *codec, OH_AVFormat *format,
                                        void *userdata)
{
    AVCodecContext *avctx = userdata;
    OHCodecDecContext *s = avctx->priv_data;

    int32_t n;

    if (OH_AVFormat_GetIntValue(format, OH_MD_KEY_WIDTH, &n))
        s->width = n;
    else
        goto out;

    if (OH_AVFormat_GetIntValue(format, OH_MD_KEY_HEIGHT, &n))
        s->height = n;
    else
        goto out;

    if (OH_AVFormat_GetIntValue(format, OH_MD_KEY_VIDEO_STRIDE, &n))
        s->stride = n;
    else
        goto out;

    if (OH_AVFormat_GetIntValue(format, OH_MD_KEY_VIDEO_SLICE_HEIGHT, &n))
        s->slice_height = n;
    else
        s->slice_height = s->height;

    if (OH_AVFormat_GetIntValue(format, OH_MD_KEY_PIXEL_FORMAT, &n)) {
        s->pix_fmt = n;
        if (s->output_to_window)
            avctx->pix_fmt = AV_PIX_FMT_OPENHARMONY;
        else
            avctx->pix_fmt = oh_pix_to_av_pix(s->pix_fmt);
        // Check whether this pixel format is supported
        if (avctx->pix_fmt == AV_PIX_FMT_NONE) {
            av_log(avctx, AV_LOG_ERROR, "Unsupported OH_AVPixelFormat %d\n",
                   n);
            goto out;
        }
    } else {
        av_log(avctx, AV_LOG_ERROR, "Failed to get pixel format\n");
        goto out;
    }

    ff_set_dimensions(avctx, s->width, s->height);
    return;
out:
    oh_decode_on_err(codec, AV_ERR_UNKNOWN, userdata);
}

static void oh_decode_on_need_input(OH_AVCodec *codec, uint32_t index, OH_AVBuffer *buffer, void *userdata)
{
    AVCodecContext *avctx = userdata;
    OHCodecDecContext *s = avctx->priv_data;
    OHBufferQueueItem item = {
        index, buffer,
    };
    
    av_log(avctx, AV_LOG_DEBUG, "on input index %u\n", index);
    ff_mutex_lock(&s->input_mutex);
    int ret = av_fifo_write(s->input_queue, &item, 1);
    if (ret >= 0)
        ff_cond_signal(&s->input_cond);
    ff_mutex_unlock(&s->input_mutex);

    if (ret < 0)
        oh_decode_on_err(codec, AV_ERR_NO_MEMORY, userdata);
}

static void oh_decode_on_output(OH_AVCodec *codec, uint32_t index, OH_AVBuffer *buffer, void *userdata)
{
    AVCodecContext *avctx = userdata;
    OHCodecDecContext *s = avctx->priv_data;
    OHBufferQueueItem item = {
        index, buffer,
    };

    av_log(avctx, AV_LOG_DEBUG, "on output index %u\n", index);
    ff_mutex_lock(&s->output_mutex);
    int ret = av_fifo_write(s->output_queue, &item, 1);
    if (ret >= 0)
        ff_cond_signal(&s->output_cond);
    ff_mutex_unlock(&s->output_mutex);

    if (ret < 0)
        oh_decode_on_err(codec, AV_ERR_NO_MEMORY, userdata);
}

static int oh_decode_start(OHCodecDecContext *s, AVCodecContext *avctx)
{
    int ret;
    OH_AVErrCode err;
    OH_AVCodecCallback cb = {
        .onError = oh_decode_on_err,
        .onStreamChanged = oh_decode_on_stream_changed,
        .onNeedInputBuffer = oh_decode_on_need_input,
        .onNewOutputBuffer = oh_decode_on_output,
    };

    err = OH_VideoDecoder_RegisterCallback(s->dec, cb, avctx);
    if (err != AV_ERR_OK) {
        ret = oh_err_to_averr(err);
        av_log(avctx, AV_LOG_ERROR, "register callback failed, %d, %s\n",
               err, av_err2str(ret));
        return ret;
    }

    err = OH_VideoDecoder_Prepare(s->dec);
    if (err != AV_ERR_OK) {
        ret = oh_err_to_averr(err);
        av_log(avctx, AV_LOG_ERROR, "prepare failed, %d, %s\n",
               err, av_err2str(ret));
        return ret;
    }
    err = OH_VideoDecoder_Start(s->dec);
    if (err != AV_ERR_OK) {
        ret = oh_err_to_averr(err);
        av_log(avctx, AV_LOG_ERROR, "start failed, %d, %s\n",
               err, av_err2str(ret));
        return ret;
    }

    return 0;
}

static av_cold int oh_decode_init(AVCodecContext *avctx)
{
    OHCodecDecContext *s = avctx->priv_data;

    ff_mutex_init(&s->input_mutex, NULL);
    ff_cond_init(&s->input_cond, NULL);
    ff_mutex_init(&s->output_mutex, NULL);
    ff_cond_init(&s->output_cond, NULL);

    int ret = oh_decode_create(s, avctx);
    if (ret < 0)
        return ret;
    ret = oh_decode_set_format(s, avctx);
    if (ret < 0)
        return ret;

    size_t fifo_size = 16;
    s->input_queue = av_fifo_alloc2(fifo_size, sizeof(OHBufferQueueItem),
                                    AV_FIFO_FLAG_AUTO_GROW);
    s->output_queue = av_fifo_alloc2(fifo_size, sizeof(OHBufferQueueItem),
                                     AV_FIFO_FLAG_AUTO_GROW);
    if (!s->input_queue || !s->output_queue)
        return AVERROR(ENOMEM);

    ret = oh_decode_start(s, avctx);
    if (ret < 0)
        return ret;

    return 0;
}

static av_cold int oh_decode_close(AVCodecContext *avctx)
{
    OHCodecDecContext *s = avctx->priv_data;

    if (s->dec) {
        OH_VideoDecoder_Stop(s->dec);
        s->dec = NULL;
        av_buffer_unref(&s->dec_ref);
    }

    av_packet_unref(&s->pkt);
    av_fifo_freep2(&s->input_queue);
    av_fifo_freep2(&s->output_queue);
    ff_mutex_destroy(&s->input_mutex);
    ff_cond_destroy(&s->input_cond);
    ff_mutex_destroy(&s->output_mutex);
    ff_cond_destroy(&s->output_cond);

    return 0;
}

static void oh_buffer_release(void *opaque, uint8_t *data)
{
    if (!opaque)
        return;

    OHCodecBuffer *buffer = opaque;

    if (!buffer->dec_ref) {
        av_free(buffer);
        return;
    }

    if (buffer->buffer) {
        OH_AVCodec *dec = (OH_AVCodec *)buffer->dec_ref->data;
        OH_AVCodecBufferAttr attr;
        OH_AVErrCode err = OH_AVBuffer_GetBufferAttr(buffer->buffer, &attr);
        if (err == AV_ERR_OK && !(attr.flags & AVCODEC_BUFFER_FLAGS_DISCARD))
            OH_VideoDecoder_RenderOutputBuffer(dec, buffer->index);
        else
            OH_VideoDecoder_FreeOutputBuffer(dec, buffer->index);
    }

    av_buffer_unref(&buffer->dec_ref);
    av_free(buffer);
}

static int oh_decode_wrap_hw_buffer(AVCodecContext *avctx, AVFrame *frame,
                                    OHBufferQueueItem *output,
                                    const OH_AVCodecBufferAttr *attr)
{
    OHCodecDecContext *s = avctx->priv_data;

    frame->format = AV_PIX_FMT_OPENHARMONY;
    OHCodecBuffer *buffer = av_mallocz(sizeof(*buffer));
    if (!buffer)
        return AVERROR(ENOMEM);

    buffer->dec_ref = av_buffer_ref(s->dec_ref);
    if (!buffer->dec_ref) {
        oh_buffer_release(buffer, NULL);
        return AVERROR(ENOMEM);
    }

    buffer->index = output->index;
    buffer->buffer = output->buffer;
    frame->buf[0] = av_buffer_create((uint8_t *)buffer->buffer, 1,
                                     oh_buffer_release,
                                     buffer, AV_BUFFER_FLAG_READONLY);
    if (!frame->buf[0]) {
        oh_buffer_release(buffer, NULL);
        return AVERROR(ENOMEM);
    }
    frame->data[3] = frame->buf[0]->data;
    frame->pts = av_rescale_q(attr->pts, AV_TIME_BASE_Q, avctx->pkt_timebase);

    return 0;
}

static int oh_decode_wrap_sw_buffer(AVCodecContext *avctx, AVFrame *frame,
                                    OHBufferQueueItem *output,
                                    const OH_AVCodecBufferAttr *attr)
{
    OHCodecDecContext *s = avctx->priv_data;

    frame->format = avctx->pix_fmt;
    if (frame->format == AV_PIX_FMT_NONE) {
        av_log(avctx, AV_LOG_ERROR, "Unsupported OH_AVPixelFormat %d\n", s->pix_fmt);
        return AVERROR_EXTERNAL;
    }

    if (!s->stride || !s->slice_height) {
        av_log(avctx, AV_LOG_ERROR,
               "buffer stride (%d) or slice height (%d) is unknown\n",
               s->stride, s->slice_height);
        return AVERROR_EXTERNAL;
    }

    int ret = ff_get_buffer(avctx, frame, 0);
    if (ret < 0)
        return ret;
    frame->pts = av_rescale_q(attr->pts, AV_TIME_BASE_Q, avctx->pkt_timebase);
    frame->pkt_dts = AV_NOPTS_VALUE;

    uint8_t *p = OH_AVBuffer_GetAddr(output->buffer);
    if (!p) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get output buffer addr\n");
        return AVERROR_EXTERNAL;
    }
    
    uint8_t *src[4] = {0};
    int src_linesizes[4] = {0};
    ret = av_image_fill_linesizes(src_linesizes, frame->format, s->stride);
    if (ret < 0)
        return ret;
    ret = av_image_fill_pointers(src, frame->format, s->slice_height, p,
                                 src_linesizes);
    if (ret < 0)
        return ret;
    av_image_copy2(frame->data, frame->linesize, src, src_linesizes,
                   frame->format, frame->width, frame->height);
    OH_AVErrCode err = OH_VideoDecoder_FreeOutputBuffer(s->dec, output->index);
    if (err != AV_ERR_OK) {
        ret = oh_err_to_averr(err);
        av_log(avctx, AV_LOG_ERROR, "RenderOutputBuffer failed, %d, %s\n", err,
               av_err2str(ret));
        return ret;
    }
    
    return 0;
}

static int oh_decode_output_frame(AVCodecContext *avctx, AVFrame *frame,
                                  OHBufferQueueItem *output)
{
    OHCodecDecContext *s = avctx->priv_data;
    OH_AVCodecBufferAttr attr;

    OH_AVErrCode err = OH_AVBuffer_GetBufferAttr(output->buffer, &attr);
    if (err != AV_ERR_OK)
        return oh_err_to_averr(err);

    av_log(avctx, AV_LOG_DEBUG, "output buffer index %u\n", output->index);
    if (attr.flags & AVCODEC_BUFFER_FLAGS_EOS) {
        av_log(avctx, AV_LOG_DEBUG, "buffer flag eos\n");
        return AVERROR_EOF;
    }
    
    frame->width = s->width;
    frame->height = s->height;

    if (s->output_to_window)
        return oh_decode_wrap_hw_buffer(avctx, frame, output, &attr);
    return oh_decode_wrap_sw_buffer(avctx, frame, output, &attr);
}

static int oh_decode_send_pkt(AVCodecContext *avctx, OHBufferQueueItem *input)
{
    OHCodecDecContext *s = avctx->priv_data;

    if (!s->pkt.size && !s->eof_sent) {
        OH_AVCodecBufferAttr attr = {
            .flags = AVCODEC_BUFFER_FLAGS_EOS,
        };
        OH_AVBuffer_SetBufferAttr(input->buffer, &attr);
        OH_VideoDecoder_PushInputBuffer(s->dec, input->index);
        s->eof_sent = true;
        return 0;
    }

    uint8_t *p = OH_AVBuffer_GetAddr(input->buffer);
    int32_t n = OH_AVBuffer_GetCapacity(input->buffer);
    if (!p || n <= 0) {
        av_log(avctx, AV_LOG_ERROR,
               "Failed to get buffer addr (%p) or capacity (%d)\n",
               p, n);
        return AVERROR_EXTERNAL;
    }
    n = FFMIN(s->pkt.size, n);
    memcpy(p, s->pkt.data, n);

    OH_AVCodecBufferAttr attr = {
            .size = n,
            .offset = 0,
            .pts = av_rescale_q(s->pkt.pts, avctx->pkt_timebase,
                                AV_TIME_BASE_Q),
            .flags = (s->pkt.flags & AV_PKT_FLAG_KEY)
                     ? AVCODEC_BUFFER_FLAGS_SYNC_FRAME : 0,
    };

    int ret;
    OH_AVErrCode err = OH_AVBuffer_SetBufferAttr(input->buffer, &attr);
    if (err != AV_ERR_OK) {
        ret = oh_err_to_averr(err);
        return ret;
    }
    err = OH_VideoDecoder_PushInputBuffer(s->dec, input->index);
    if (err != AV_ERR_OK) {
        ret = oh_err_to_averr(err);
        av_log(avctx, AV_LOG_ERROR, "push input buffer failed, %d, %s\n",
               err, av_err2str(ret));
        return ret;
    }

    if (n < s->pkt.size) {
        s->pkt.size -= n;
        s->pkt.data += n;
    } else {
        av_packet_unref(&s->pkt);
    }
    av_log(avctx, AV_LOG_DEBUG, "queue input buffer index %u\n", input->index);

    return 0;
}

static int oh_decode_receive_frame(AVCodecContext *avctx, AVFrame *frame)
{
    OHCodecDecContext *s = avctx->priv_data;

    while (1) {
        OHBufferQueueItem buffer = {0};
        int ret;

        // Try get output
        ff_mutex_lock(&s->output_mutex);
        while (!s->decode_status) {
            if (av_fifo_read(s->output_queue, &buffer, 1) >= 0)
                break;
            // Only wait after send EOF
            if (s->eof_sent && !s->decode_status)
                ff_cond_wait(&s->output_cond, &s->output_mutex);
            else
                break;
        }

        ret = s->decode_status;
        ff_mutex_unlock(&s->output_mutex);

        if (buffer.buffer) {
            ret = oh_decode_output_frame(avctx, frame, &buffer);
            return ret;
        }
        if (ret < 0)
            return ret;

        if (!s->pkt.size) {
            /* fetch new packet or eof */
            ret = ff_decode_get_packet(avctx, &s->pkt);
            if (ret < 0 && ret != AVERROR_EOF)
                return ret;
        }

        // Wait input buffer
        ff_mutex_lock(&s->input_mutex);
        while (!s->decode_status) {
            if (av_fifo_read(s->input_queue, &buffer, 1) >= 0)
                break;
            ff_cond_wait(&s->input_cond, &s->input_mutex);
        }

        ret = s->decode_status;
        ff_mutex_unlock(&s->input_mutex);

        if (ret < 0)
            return ret;

        ret = oh_decode_send_pkt(avctx, &buffer);
        if (ret < 0)
            return ret;
    }

    return AVERROR(EAGAIN);
}

static void oh_decode_flush(AVCodecContext *avctx)
{
    OHCodecDecContext *s = avctx->priv_data;

    OH_VideoDecoder_Flush(s->dec);

    ff_mutex_lock(&s->input_mutex);
    av_fifo_reset2(s->input_queue);
    ff_mutex_unlock(&s->input_mutex);

    ff_mutex_lock(&s->output_mutex);
    av_fifo_reset2(s->output_queue);
    ff_mutex_unlock(&s->output_mutex);

    OH_VideoDecoder_Start(s->dec);
}

static const AVCodecHWConfigInternal *const oh_hw_configs[] = {
    &(const AVCodecHWConfigInternal) {
        .public = {
            .pix_fmt = AV_PIX_FMT_OPENHARMONY,
            .methods = AV_CODEC_HW_CONFIG_METHOD_AD_HOC |
                           AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX,
            .device_type = AV_HWDEVICE_TYPE_OPENHARMONY,
        },
        .hwaccel = NULL,
    },
    NULL
};

#define OFFSET(x) offsetof(OHCodecDecContext, x)
#define VD AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM
static const AVOption ohcodec_vdec_options[] = {
    { NULL }
};

#define DECLARE_OHCODEC_VCLASS(short_name)                                     \
  static const AVClass short_name##_oh_dec_class = {                           \
      .class_name = #short_name "_ohcodec",                                    \
      .item_name = av_default_item_name,                                       \
      .option = ohcodec_vdec_options,                                          \
      .version = LIBAVUTIL_VERSION_INT,                                        \
  };

#define DECLARE_OHCODEC_VDEC(short_name, full_name, codec_id, bsf)             \
  DECLARE_OHCODEC_VCLASS(short_name)                                           \
  const FFCodec ff_##short_name##_oh_decoder = {                               \
      .p.name = #short_name "_ohcodec",                                        \
      CODEC_LONG_NAME(full_name " OpenHarmony Codec"),                         \
      .p.type = AVMEDIA_TYPE_VIDEO,                                            \
      .p.id = codec_id,                                                        \
      .p.priv_class = &short_name##_oh_dec_class,                              \
      .priv_data_size = sizeof(OHCodecDecContext),                             \
      .init = oh_decode_init,                                                  \
      FF_CODEC_RECEIVE_FRAME_CB(oh_decode_receive_frame),                             \
      .flush = oh_decode_flush,                                                \
      .close = oh_decode_close,                                                \
      .p.capabilities = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_AVOID_PROBING |      \
                        AV_CODEC_CAP_HARDWARE,                                 \
      .caps_internal = FF_CODEC_CAP_INIT_CLEANUP,                              \
      .bsfs = bsf,                                                             \
      .hw_configs = oh_hw_configs,                                             \
      .p.wrapper_name = "ohcodec",                                             \
  };

#if CONFIG_H264_OH_DECODER
DECLARE_OHCODEC_VDEC(h264, "H.264", AV_CODEC_ID_H264, "h264_mp4toannexb")
#endif
