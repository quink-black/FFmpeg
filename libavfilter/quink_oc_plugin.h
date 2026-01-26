/*
 * OpenCV Plugin Interface for FFmpeg
 *
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

#ifndef AVFILTER_QUINK_OC_PLUGIN_H
#define AVFILTER_QUINK_OC_PLUGIN_H

#include <opencv2/core.hpp>
#include <vector>

#define QUINK_OC_PLUGIN_API_VERSION 1

#if defined(_WIN32) || defined(_WIN64)
    #define QUINK_OC_EXPORT __declspec(dllexport)
#else
    #define QUINK_OC_EXPORT __attribute__((visibility("default")))
#endif

enum QuinkOCProcessResult {
    OK = 0,              ///< Success, output frame(s) produced
    TRY_AGAIN = 1,       ///< Success, but output not ready yet
    ERROR = -1           ///< Processing error
};

class QuinkOCPlugin {
public:
    virtual ~QuinkOCPlugin() = default;

    /**
     * Initialize the plugin
     * @param params  User-specified parameter string (may be NULL)
     * @return true on success
     */
    virtual bool init(const char *params) = 0;

    /**
     * Process frames
     *
     * This method is called for each set of input frames.
     * @param inputs   Input cv::Mat images (zero-copy from FFmpeg, refcount tied to AVFrame)
     * @param outputs  Output cv::Mat images (pre-allocated buffer to write into)
     */
    virtual QuinkOCProcessResult process(const std::vector<cv::Mat> &inputs,
                                         std::vector<cv::Mat> &outputs) = 0;

    /**
     * Flush buffered frames at end of stream
     *
     * Called when input stream ends. The plugin should output any remaining
     * buffered frames. This method may be called multiple times until it
     * returns false (no more frames to output).
     *
     * @param outputs  Output buffer to write flushed frame into
     * @return true if a frame was output, false if no more frames
     */
    virtual bool flush(std::vector<cv::Mat> &outputs) = 0;

    virtual int numInputs() const = 0;

    virtual int numOutputs() const = 0;

    /**
     * Configure plugin with input dimensions
     *
     * Called during filter configuration. Plugin can adjust output dimensions.
     *
     * @param input_idx   Input index (0-based)
     * @param width       Input width
     * @param height      Input height
     * @param cv_type     OpenCV type (e.g., CV_8UC3)
     * @param out_width   [in/out] Output width (initialized to input width)
     * @param out_height  [in/out] Output height (initialized to input height)
     * @return true on success
     */
    virtual bool configure(int input_idx, int width, int height, int cv_type,
                           int &out_width, int &out_height) = 0;
    virtual void uninit() = 0;
};

/**
 * Plugin Descriptor
 *
 * Contains plugin metadata and factory functions.
 * Plugins export a single function that returns a pointer to a static descriptor.
 */
struct QuinkOCPluginDescriptor {
    int api_version;            ///< Must be QUINK_OC_PLUGIN_API_VERSION
    const char *name;           ///< Plugin name
    const char *description;    ///< Plugin description

    QuinkOCPlugin* (*create)();            ///< Create plugin instance
    void (*destroy)(QuinkOCPlugin* p);     ///< Destroy plugin instance
};

typedef const QuinkOCPluginDescriptor* (*QuinkOCPluginGetDescriptorFunc)();

/** Symbol name to load from shared library */
#define QUINK_OC_PLUGIN_DESCRIPTOR_SYMBOL "quink_oc_plugin_get_descriptor"

/**
 * Plugin entry macro
 *
 * Usage: QUINK_OC_PLUGIN_ENTRY(PluginClass, "name", "description")
 */
#define QUINK_OC_PLUGIN_ENTRY(PluginClass, plugin_name, plugin_desc) \
    static QuinkOCPlugin* _quink_create() { return new PluginClass(); } \
    static void _quink_destroy(QuinkOCPlugin* p) { delete p; } \
    extern "C" QUINK_OC_EXPORT const QuinkOCPluginDescriptor* quink_oc_plugin_get_descriptor() { \
        static const QuinkOCPluginDescriptor desc = { \
            QUINK_OC_PLUGIN_API_VERSION, \
            plugin_name, \
            plugin_desc, \
            _quink_create, \
            _quink_destroy \
        }; \
        return &desc; \
    }

#endif /* AVFILTER_QUINK_OC_PLUGIN_H */
