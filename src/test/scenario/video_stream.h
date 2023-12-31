/*
 *  Copyright 2018 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef TEST_SCENARIO_VIDEO_STREAM_H_
#define TEST_SCENARIO_VIDEO_STREAM_H_
#include <memory>
#include <string>
#include <vector>

#include "rtc_base/constructor_magic.h"
#include "test/fake_encoder.h"
#include "test/frame_generator_capturer.h"
#include "test/logging/log_writer.h"
#include "test/scenario/call_client.h"
#include "test/scenario/column_printer.h"
#include "test/scenario/network_node.h"
#include "test/scenario/quality_stats.h"
#include "test/scenario/scenario_config.h"
#include "test/test_video_capturer.h"

namespace webrtc {
namespace test {
// SendVideoStream provides an interface for changing parameters and retrieving
// states at run time.
class SendVideoStream {
 public:
  RTC_DISALLOW_COPY_AND_ASSIGN(SendVideoStream);
  ~SendVideoStream();
  void SetCaptureFramerate(int framerate);
  VideoSendStream::Stats GetStats() const;
  ColumnPrinter StatsPrinter();
  void Start();
  void Stop();
  void UpdateConfig(std::function<void(VideoStreamConfig*)> modifier);

 private:
  friend class Scenario;
  friend class VideoStreamPair;
  friend class ReceiveVideoStream;
  // Handles RTCP feedback for this stream.
  SendVideoStream(CallClient* sender,
                  VideoStreamConfig config,
                  Transport* send_transport,
                  VideoQualityAnalyzer* analyzer);

  rtc::CriticalSection crit_;
  std::vector<uint32_t> ssrcs_;
  std::vector<uint32_t> rtx_ssrcs_;
  VideoSendStream* send_stream_ = nullptr;
  CallClient* const sender_;
  VideoStreamConfig config_ RTC_GUARDED_BY(crit_);
  std::unique_ptr<VideoEncoderFactory> encoder_factory_;
  std::vector<test::FakeEncoder*> fake_encoders_ RTC_GUARDED_BY(crit_);
  std::unique_ptr<VideoBitrateAllocatorFactory> bitrate_allocator_factory_;
  std::unique_ptr<FrameGeneratorCapturer> video_capturer_;
  std::unique_ptr<ForwardingCapturedFrameTap> frame_tap_;
  int next_local_network_id_ = 0;
  int next_remote_network_id_ = 0;
};

// ReceiveVideoStream represents a video receiver. It can't be used directly.
class ReceiveVideoStream {
 public:
  RTC_DISALLOW_COPY_AND_ASSIGN(ReceiveVideoStream);
  ~ReceiveVideoStream();
  void Start();
  void Stop();

 private:
  friend class Scenario;
  friend class VideoStreamPair;
  ReceiveVideoStream(CallClient* receiver,
                     VideoStreamConfig config,
                     SendVideoStream* send_stream,
                     size_t chosen_stream,
                     Transport* feedback_transport,
                     VideoQualityAnalyzer* analyzer);

  VideoReceiveStream* receive_stream_ = nullptr;
  FlexfecReceiveStream* flecfec_stream_ = nullptr;
  std::unique_ptr<rtc::VideoSinkInterface<VideoFrame>> renderer_;
  CallClient* const receiver_;
  const VideoStreamConfig config_;
  std::unique_ptr<VideoDecoderFactory> decoder_factory_;
};

// VideoStreamPair represents a video streaming session. It can be used to
// access underlying send and receive classes. It can also be used in calls to
// the Scenario class.
class VideoStreamPair {
 public:
  RTC_DISALLOW_COPY_AND_ASSIGN(VideoStreamPair);
  ~VideoStreamPair();
  SendVideoStream* send() { return &send_stream_; }
  ReceiveVideoStream* receive() { return &receive_stream_; }
  VideoQualityAnalyzer* analyzer() { return &analyzer_; }

 private:
  friend class Scenario;
  VideoStreamPair(CallClient* sender,
                  CallClient* receiver,
                  VideoStreamConfig config,
                  std::unique_ptr<RtcEventLogOutput> quality_writer);

  const VideoStreamConfig config_;

  VideoQualityAnalyzer analyzer_;
  SendVideoStream send_stream_;
  ReceiveVideoStream receive_stream_;
};
}  // namespace test
}  // namespace webrtc

#endif  // TEST_SCENARIO_VIDEO_STREAM_H_
