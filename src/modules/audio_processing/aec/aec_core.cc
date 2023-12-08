/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/*
 * The core AEC algorithm, which is presented with time-aligned signals.
 */
#include <math.h>
#include <stddef.h>  // size_t
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include "modules/audio_processing/aec/aec_core.h"
#include "rtc_base/checks.h"

extern "C" {
#include "common_audio/ring_buffer.h"
}
#include "common_audio/signal_processing/include/signal_processing_library.h"
#include "modules/audio_processing/aec/aec_common.h"
#include "modules/audio_processing/aec/aec_core_optimized_methods.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"
#include "modules/audio_processing/utility/delay_estimator_wrapper.h"
#include "rtc_base/system/arch.h"
#include "system_wrappers/include/cpu_features_wrapper.h"
#include "system_wrappers/include/metrics.h"

namespace webrtc {
namespace {
enum class DelaySource {
  kSystemDelay,    // The delay values come from the OS.
  kDelayAgnostic,  // The delay values come from the DA-AEC.
};

constexpr int kMinDelayLogValue = -200;
constexpr int kMaxDelayLogValue = 200;
constexpr int kNumDelayLogBuckets = 100;

void MaybeLogDelayAdjustment(int moved_ms, DelaySource source) {
  if (moved_ms == 0)
    return;
  switch (source) {
    case DelaySource::kSystemDelay:
      RTC_HISTOGRAM_COUNTS("WebRTC.Audio.AecDelayAdjustmentMsSystemValue",
                           moved_ms, kMinDelayLogValue, kMaxDelayLogValue,
                           kNumDelayLogBuckets);
      return;
    case DelaySource::kDelayAgnostic:
      RTC_HISTOGRAM_COUNTS("WebRTC.Audio.AecDelayAdjustmentMsAgnosticValue",
                           moved_ms, kMinDelayLogValue, kMaxDelayLogValue,
                           kNumDelayLogBuckets);
      return;
  }
}
}  // namespace

// Buffer size (samples)
static const size_t kBufferSizeBlocks = 250;  // 1 second of audio in 16 kHz.

// Metrics
static const size_t kSubCountLen = 4;
static const size_t kCountLen = 50;
static const int kDelayMetricsAggregationWindow = 1250;  // 5 seconds at 16 kHz.

// Divergence metric is based on audio level, which gets updated every
// |kSubCountLen + 1| * PART_LEN samples. Divergence metric takes the statistics
// of |kDivergentFilterFractionAggregationWindowSize| audio levels. The
// following value corresponds to 1 second at 16 kHz.
static const int kDivergentFilterFractionAggregationWindowSize = 50;

// Quantities to control H band scaling for SWB input
static const float cnScaleHband = 0.4f;  // scale for comfort noise in H band.
// Initial bin for averaging nlp gain in low band
static const int freqAvgIc = PART_LEN / 2;

// Matlab code to produce table:
// win = sqrt(hanning(63)); win = [0 ; win(1:32)];
// fprintf(1, '\t%.14f, %.14f, %.14f,\n', win);
ALIGN16_BEG const float ALIGN16_END WebRtcAec_sqrtHanning[65] = {
    0.00000000000000f, 0.02454122852291f, 0.04906767432742f, 0.07356456359967f,
    0.09801714032956f, 0.12241067519922f, 0.14673047445536f, 0.17096188876030f,
    0.19509032201613f, 0.21910124015687f, 0.24298017990326f, 0.26671275747490f,
    0.29028467725446f, 0.31368174039889f, 0.33688985339222f, 0.35989503653499f,
    0.38268343236509f, 0.40524131400499f, 0.42755509343028f, 0.44961132965461f,
    0.47139673682600f, 0.49289819222978f, 0.51410274419322f, 0.53499761988710f,
    0.55557023301960f, 0.57580819141785f, 0.59569930449243f, 0.61523159058063f,
    0.63439328416365f, 0.65317284295378f, 0.67155895484702f, 0.68954054473707f,
    0.70710678118655f, 0.72424708295147f, 0.74095112535496f, 0.75720884650648f,
    0.77301045336274f, 0.78834642762661f, 0.80320753148064f, 0.81758481315158f,
    0.83146961230255f, 0.84485356524971f, 0.85772861000027f, 0.87008699110871f,
    0.88192126434835f, 0.89322430119552f, 0.90398929312344f, 0.91420975570353f,
    0.92387953251129f, 0.93299279883474f, 0.94154406518302f, 0.94952818059304f,
    0.95694033573221f, 0.96377606579544f, 0.97003125319454f, 0.97570213003853f,
    0.98078528040323f, 0.98527764238894f, 0.98917650996478f, 0.99247953459871f,
    0.99518472667220f, 0.99729045667869f, 0.99879545620517f, 0.99969881869620f,
    1.00000000000000f};

// Matlab code to produce table:
// weightCurve = [0 ; 0.3 * sqrt(linspace(0,1,64))' + 0.1];
// fprintf(1, '\t%.4f, %.4f, %.4f, %.4f, %.4f, %.4f,\n', weightCurve);
ALIGN16_BEG const float ALIGN16_END WebRtcAec_weightCurve[65] = {
    0.0000f, 0.1000f, 0.1378f, 0.1535f, 0.1655f, 0.1756f, 0.1845f, 0.1926f,
    0.2000f, 0.2069f, 0.2134f, 0.2195f, 0.2254f, 0.2309f, 0.2363f, 0.2414f,
    0.2464f, 0.2512f, 0.2558f, 0.2604f, 0.2648f, 0.2690f, 0.2732f, 0.2773f,
    0.2813f, 0.2852f, 0.2890f, 0.2927f, 0.2964f, 0.3000f, 0.3035f, 0.3070f,
    0.3104f, 0.3138f, 0.3171f, 0.3204f, 0.3236f, 0.3268f, 0.3299f, 0.3330f,
    0.3360f, 0.3390f, 0.3420f, 0.3449f, 0.3478f, 0.3507f, 0.3535f, 0.3563f,
    0.3591f, 0.3619f, 0.3646f, 0.3673f, 0.3699f, 0.3726f, 0.3752f, 0.3777f,
    0.3803f, 0.3828f, 0.3854f, 0.3878f, 0.3903f, 0.3928f, 0.3952f, 0.3976f,
    0.4000f};

// Matlab code to produce table:
// overDriveCurve = [sqrt(linspace(0,1,65))' + 1];
// fprintf(1, '\t%.4f, %.4f, %.4f, %.4f, %.4f, %.4f,\n', overDriveCurve);
ALIGN16_BEG const float ALIGN16_END WebRtcAec_overDriveCurve[65] = {
    1.0000f, 1.1250f, 1.1768f, 1.2165f, 1.2500f, 1.2795f, 1.3062f, 1.3307f,
    1.3536f, 1.3750f, 1.3953f, 1.4146f, 1.4330f, 1.4507f, 1.4677f, 1.4841f,
    1.5000f, 1.5154f, 1.5303f, 1.5449f, 1.5590f, 1.5728f, 1.5863f, 1.5995f,
    1.6124f, 1.6250f, 1.6374f, 1.6495f, 1.6614f, 1.6731f, 1.6847f, 1.6960f,
    1.7071f, 1.7181f, 1.7289f, 1.7395f, 1.7500f, 1.7603f, 1.7706f, 1.7806f,
    1.7906f, 1.8004f, 1.8101f, 1.8197f, 1.8292f, 1.8385f, 1.8478f, 1.8570f,
    1.8660f, 1.8750f, 1.8839f, 1.8927f, 1.9014f, 1.9100f, 1.9186f, 1.9270f,
    1.9354f, 1.9437f, 1.9520f, 1.9601f, 1.9682f, 1.9763f, 1.9843f, 1.9922f,
    2.0000f};

// Delay Agnostic AEC parameters, still under development and may change.
static const float kDelayQualityThresholdMax = 0.07f;
static const float kDelayQualityThresholdMin = 0.01f;
static const int kInitialShiftOffset = 5;
#if !defined(WEBRTC_ANDROID)
static const int kDelayCorrectionStart = 1500;  // 10 ms chunks
#endif

// Target suppression levels for nlp modes.
// log{0.001, 0.00001, 0.00000001}
static const float kTargetSupp[3] = {-6.9f, -11.5f, -18.4f};

// Two sets of parameters, one for the extended filter mode.
static const float kExtendedMinOverDrive[3] = {3.0f, 6.0f, 15.0f};
static const float kNormalMinOverDrive[3] = {1.0f, 2.0f, 5.0f};
const float WebRtcAec_kExtendedSmoothingCoefficients[2][2] = {{0.9f, 0.1f},
                                                              {0.92f, 0.08f}};
const float WebRtcAec_kNormalSmoothingCoefficients[2][2] = {{0.9f, 0.1f},
                                                            {0.93f, 0.07f}};

// Number of partitions forming the NLP's "preferred" bands.
enum { kPrefBandSize = 24 };

#ifdef AUMDF_FILTER
ALIGN16_BEG const float ALIGN16_END WebRtcAec_mdfWindows[128] = {
    0.00000000000000000f, 0.000602271876870619f, 0.00240763658182785f,
    0.00541174483331447f, 0.00960735947167074f,  0.0149843728940851f,
    0.0215298314046942f,  0.0292279664211693f,   0.0380602324626113f,
    0.0480053518272369f,  0.0590393658522246f,   0.0711356926322292f,
    0.0842651910575186f,  0.0983962310174570f,   0.113494769600209f,
    0.129524433105094f,   0.146446604670013f,    0.164220517302849f,
    0.182803352092721f,   0.202150341364480f,    0.222214876527967f,
    0.242948620362175f,   0.264301623463849f,    0.286222444579970f,
    0.308658274534234f,   0.331555063448978f,    0.354857650956065f,
    0.378509899083041f,   0.402454827494418f,    0.426634750762294f,
    0.450991417335592f,   0.475466149873147f,    0.499999986602552f,
    0.524533823364233f,   0.549008555998536f,    0.573365222732823f,
    0.597545146225541f,   0.621490074925072f,    0.645142323402818f,
    0.668444911322446f,   0.691341700710509f,    0.713777531197730f,
    0.735698352905162f,   0.757051356655076f,    0.777785101192891f,
    0.797849637113657f,   0.817196627194544f,    0.835779462843442f,
    0.853553376383134f,   0.870475548900553f,    0.886505213401288f,
    0.901603753020841f,   0.915734794056034f,    0.928864293592441f,
    0.940960621516748f,   0.951994636716468f,    0.961939757283425f,
    0.970772024551901f,   0.978470160817157f,    0.985015620595286f,
    0.990392635300904f,   0.994588251235057f,    0.997592360791812f,
    0.999397726808365f,   0.999999999999999f,    0.999397729437892f,
    0.997592366044531f,   0.994588259098313f,    0.990392645755754f,
    0.985015633616543f,   0.978470176373453f,    0.970772042605759f,
    0.961939777791351f,   0.951994659629057f,    0.940960646778801f,
    0.928864321143100f,   0.915734823828928f,    0.901603784944244f,
    0.886505247398293f,   0.870475584889258f,    0.853553414276839f,
    0.835779502550859f,   0.817196668620014f,    0.797849680157382f,
    0.777785145751175f,   0.757051402620574f,    0.735698400167140f,
    0.713777579642329f,   0.691341750221022f,    0.668444961779598f,
    0.645142374685052f,   0.621490126908846f,    0.597545198785622f,
    0.573365275742588f,   0.549008609330280f,    0.524533876889474f,
    0.500000040192345f,   0.475466203398389f,    0.450991470667336f,
    0.426634803772059f,   0.402454880054499f,    0.378509951066816f,
    0.354857702238300f,   0.331555113906130f,    0.308658324044748f,
    0.286222493024570f,   0.264301670725828f,    0.242948666327674f,
    0.222214921086253f,   0.202150384408207f,    0.182803393518192f,
    0.164220557010269f,   0.146446642563721f,    0.129524469093801f,
    0.113494803597216f,   0.0983962629408615f,   0.0842652208304144f,
    0.0711357201828909f,  0.0590393911142804f,   0.0480053747398281f,
    0.0380602529705396f,  0.0292279844750292f,   0.0215298469609925f,
    0.0149843859153453f,  0.00960736992652334f,  0.00541175269657296f,
    0.00240764183454889f, 0.000602274506399991f};
#endif

#ifndef AUMDF_FILTER
WebRtcAecFilterFar WebRtcAec_FilterFar;
WebRtcAecScaleErrorSignal WebRtcAec_ScaleErrorSignal;
WebRtcAecFilterAdaptation WebRtcAec_FilterAdaptation;
#endif
WebRtcAecOverdrive WebRtcAec_Overdrive;
WebRtcAecSuppress WebRtcAec_Suppress;
WebRtcAecComputeCoherence WebRtcAec_ComputeCoherence;
WebRtcAecUpdateCoherenceSpectra WebRtcAec_UpdateCoherenceSpectra;
WebRtcAecStoreAsComplex WebRtcAec_StoreAsComplex;
#ifndef AUMDF_FILTER
WebRtcAecPartitionDelay WebRtcAec_PartitionDelay;
#endif
WebRtcAecWindowData WebRtcAec_WindowData;

/*__inline static float MulRe(float aRe, float aIm, float bRe, float bIm) {
  return aRe * bRe - aIm * bIm;
}

__inline static float MulIm(float aRe, float aIm, float bRe, float bIm) {
  return aRe * bIm + aIm * bRe;
}*/

// TODO(minyue): Due to a legacy bug, |framelevel| and |averagelevel| use a
// window, of which the length is 1 unit longer than indicated. Remove "+1" when
// the code is refactored.
PowerLevel::PowerLevel()
    : framelevel(kSubCountLen + 1), averagelevel(kCountLen + 1) {}

BlockBuffer::BlockBuffer() {
  buffer_ = WebRtc_CreateBuffer(kBufferSizeBlocks, sizeof(float) * PART_LEN);
  RTC_CHECK(buffer_);
  ReInit();
}

BlockBuffer::~BlockBuffer() {
  WebRtc_FreeBuffer(buffer_);
}

void BlockBuffer::ReInit() {
  WebRtc_InitBuffer(buffer_);
}

void BlockBuffer::Insert(const float block[PART_LEN]) {
  WebRtc_WriteBuffer(buffer_, block, 1);
}

void BlockBuffer::ExtractExtendedBlock(float extended_block[PART_LEN2]) {
  float* block_ptr = NULL;
  RTC_DCHECK_LT(0, AvaliableSpace());

  // Extract the previous block.
  WebRtc_MoveReadPtr(buffer_, -1);
  size_t read_elements = WebRtc_ReadBuffer(
      buffer_, reinterpret_cast<void**>(&block_ptr), &extended_block[0], 1);
  if (read_elements == 0u) {
    std::fill_n(&extended_block[0], PART_LEN, 0.0f);
  } else if (block_ptr != &extended_block[0]) {
    memcpy(&extended_block[0], block_ptr, PART_LEN * sizeof(float));
  }

  // Extract the current block.
  read_elements =
      WebRtc_ReadBuffer(buffer_, reinterpret_cast<void**>(&block_ptr),
                        &extended_block[PART_LEN], 1);
  if (read_elements == 0u) {
    std::fill_n(&extended_block[PART_LEN], PART_LEN, 0.0f);
  } else if (block_ptr != &extended_block[PART_LEN]) {
    memcpy(&extended_block[PART_LEN], block_ptr, PART_LEN * sizeof(float));
  }
}

int BlockBuffer::AdjustSize(int buffer_size_decrease) {
  return WebRtc_MoveReadPtr(buffer_, buffer_size_decrease);
}

size_t BlockBuffer::Size() {
  return static_cast<int>(WebRtc_available_read(buffer_));
}

size_t BlockBuffer::AvaliableSpace() {
  return WebRtc_available_write(buffer_);
}

DivergentFilterFraction::DivergentFilterFraction()
    : count_(0), occurrence_(0), fraction_(-1.0) {}

void DivergentFilterFraction::Reset() {
  Clear();
  fraction_ = -1.0;
}

void DivergentFilterFraction::AddObservation(const PowerLevel& nearlevel,
                                             const PowerLevel& linoutlevel,
                                             const PowerLevel& nlpoutlevel) {
  const float near_level = nearlevel.framelevel.GetLatestMean();
  const float level_increase =
      linoutlevel.framelevel.GetLatestMean() - near_level;
  const bool output_signal_active =
      nlpoutlevel.framelevel.GetLatestMean() > 40.0 * nlpoutlevel.minlevel;
  // Level increase should be, in principle, negative, when the filter
  // does not diverge. Here we allow some margin (0.01 * near end level) and
  // numerical error (1.0). We count divergence only when the AEC output
  // signal is active.
  if (output_signal_active && level_increase > std::max(0.01 * near_level, 1.0))
    occurrence_++;
  ++count_;
  if (count_ == kDivergentFilterFractionAggregationWindowSize) {
    fraction_ = static_cast<float>(occurrence_) /
                kDivergentFilterFractionAggregationWindowSize;
    Clear();
  }
}

float DivergentFilterFraction::GetLatestFraction() const {
  return fraction_;
}

void DivergentFilterFraction::Clear() {
  count_ = 0;
  occurrence_ = 0;
}

// TODO(minyue): Moving some initialization from WebRtcAec_CreateAec() to ctor.
AecCore::AecCore(int instance_index)
    : data_dumper(new ApmDataDumper(instance_index)) {}

AecCore::~AecCore() {}

static int CmpFloat(const void* a, const void* b) {
  const float* da = (const float*)a;
  const float* db = (const float*)b;

  return (*da > *db) - (*da < *db);
}

/*static void FilterFar(int num_partitions,
                      int x_fft_buf_block_pos,
                      float x_fft_buf[2][kExtendedNumPartitions * PART_LEN1],
                      float h_fft_buf[2][kExtendedNumPartitions * PART_LEN1],
                      float y_fft[2][PART_LEN1]) {
  int i;
  for (i = 0; i < num_partitions; i++) {
    int j;
    int xPos = (i + x_fft_buf_block_pos) * PART_LEN1;
    int pos = i * PART_LEN1;
    // Check for wrap
    if (i + x_fft_buf_block_pos >= num_partitions) {
      xPos -= num_partitions * (PART_LEN1);
    }

    for (j = 0; j < PART_LEN1; j++) {
      y_fft[0][j] += MulRe(x_fft_buf[0][xPos + j], x_fft_buf[1][xPos + j],
                           h_fft_buf[0][pos + j], h_fft_buf[1][pos + j]);
      y_fft[1][j] += MulIm(x_fft_buf[0][xPos + j], x_fft_buf[1][xPos + j],
                           h_fft_buf[0][pos + j], h_fft_buf[1][pos + j]);
    }
  }
}*/

/*static void ScaleErrorSignal(float mu,
                             float error_threshold,
                             float x_pow[PART_LEN1],
                             float ef[2][PART_LEN1]) {
  int i;
  float abs_ef;
  for (i = 0; i < (PART_LEN1); i++) {
    ef[0][i] /= (x_pow[i] + 1e-10f);
    ef[1][i] /= (x_pow[i] + 1e-10f);
    abs_ef = sqrtf(ef[0][i] * ef[0][i] + ef[1][i] * ef[1][i]);

    if (abs_ef > error_threshold) {
      abs_ef = error_threshold / (abs_ef + 1e-10f);
      ef[0][i] *= abs_ef;
      ef[1][i] *= abs_ef;
    }

    // Stepsize factor
    ef[0][i] *= mu;
    ef[1][i] *= mu;
  }
}*/

/*static void FilterAdaptation(
    const OouraFft& ooura_fft,
    int num_partitions,
    int x_fft_buf_block_pos,
    float x_fft_buf[2][kExtendedNumPartitions * PART_LEN1],
    float e_fft[2][PART_LEN1],
    float h_fft_buf[2][kExtendedNumPartitions * PART_LEN1]) {
  int i, j;
  float fft[PART_LEN2];
  for (i = 0; i < num_partitions; i++) {
    int xPos = (i + x_fft_buf_block_pos) * (PART_LEN1);
    int pos;
    // Check for wrap
    if (i + x_fft_buf_block_pos >= num_partitions) {
      xPos -= num_partitions * PART_LEN1;
    }

    pos = i * PART_LEN1;

    for (j = 0; j < PART_LEN; j++) {
      fft[2 * j] = MulRe(x_fft_buf[0][xPos + j], -x_fft_buf[1][xPos + j],
                         e_fft[0][j], e_fft[1][j]);
      fft[2 * j + 1] = MulIm(x_fft_buf[0][xPos + j], -x_fft_buf[1][xPos + j],
                             e_fft[0][j], e_fft[1][j]);
    }
    fft[1] =
        MulRe(x_fft_buf[0][xPos + PART_LEN], -x_fft_buf[1][xPos + PART_LEN],
              e_fft[0][PART_LEN], e_fft[1][PART_LEN]);

    ooura_fft.InverseFft(fft);
    memset(fft + PART_LEN, 0, sizeof(float) * PART_LEN);

    // fft scaling
    {
      float scale = 2.0f / PART_LEN2;
      for (j = 0; j < PART_LEN; j++) {
        fft[j] *= scale;
      }
    }
    ooura_fft.Fft(fft);

    h_fft_buf[0][pos] += fft[0];
    h_fft_buf[0][pos + PART_LEN] += fft[1];

    for (j = 1; j < PART_LEN; j++) {
      h_fft_buf[0][pos + j] += fft[2 * j];
      h_fft_buf[1][pos + j] += fft[2 * j + 1];
    }
  }
}*/

static void Overdrive(float overdrive_scaling,
                      const float hNlFb,
                      float hNl[PART_LEN1]) {
  for (int i = 0; i < PART_LEN1; ++i) {
    // Weight subbands
    if (hNl[i] > hNlFb) {
      hNl[i] = WebRtcAec_weightCurve[i] * hNlFb +
               (1 - WebRtcAec_weightCurve[i]) * hNl[i];
    }
    hNl[i] = powf(hNl[i], overdrive_scaling * WebRtcAec_overDriveCurve[i]);
  }
}

static void Suppress(const float hNl[PART_LEN1], float efw[2][PART_LEN1]) {
  for (int i = 0; i < PART_LEN1; ++i) {
    // Suppress error signal
    efw[0][i] *= hNl[i];
    efw[1][i] *= hNl[i];

    // Ooura fft returns incorrect sign on imaginary component. It matters here
    // because we are making an additive change with comfort noise.
    efw[1][i] *= -1;
  }
}

/*static int PartitionDelay(
    int num_partitions,
    float h_fft_buf[2][kExtendedNumPartitions * PART_LEN1]) {
  // Measures the energy in each filter partition and returns the partition with
  // highest energy.
  // TODO(bjornv): Spread computational cost by computing one partition per
  // block?
  float wfEnMax = 0;
  int i;
  int delay = 0;

  for (i = 0; i < num_partitions; i++) {
    int j;
    int pos = i * PART_LEN1;
    float wfEn = 0;
    for (j = 0; j < PART_LEN1; j++) {
      wfEn += h_fft_buf[0][pos + j] * h_fft_buf[0][pos + j] +
              h_fft_buf[1][pos + j] * h_fft_buf[1][pos + j];
    }

    if (wfEn > wfEnMax) {
      wfEnMax = wfEn;
      delay = i;
    }
  }
  return delay;
}*/

// Update metric with 10 * log10(numerator / denominator).
static void UpdateLogRatioMetric(Stats* metric,
                                 float numerator,
                                 float denominator) {
  RTC_DCHECK(metric);
  RTC_CHECK(numerator >= 0);
  RTC_CHECK(denominator >= 0);

  const float log_numerator = log10(numerator + 1e-10f);
  const float log_denominator = log10(denominator + 1e-10f);
  metric->instant = 10.0f * (log_numerator - log_denominator);

  // Max.
  if (metric->instant > metric->max)
    metric->max = metric->instant;

  // Min.
  if (metric->instant < metric->min)
    metric->min = metric->instant;

  // Average.
  metric->counter++;
  // This is to protect overflow, which should almost never happen.
  RTC_CHECK_NE(0, metric->counter);
  metric->sum += metric->instant;
  metric->average = metric->sum / metric->counter;

  // Upper mean.
  if (metric->instant > metric->average) {
    metric->hicounter++;
    // This is to protect overflow, which should almost never happen.
    RTC_CHECK_NE(0, metric->hicounter);
    metric->hisum += metric->instant;
    metric->himean = metric->hisum / metric->hicounter;
  }
}

// Threshold to protect against the ill-effects of a zero far-end.
const float WebRtcAec_kMinFarendPSD = 15;

// Updates the following smoothed Power Spectral Densities (PSD):
//  - sd  : near-end
//  - se  : residual echo
//  - sx  : far-end
//  - sde : cross-PSD of near-end and residual echo
//  - sxd : cross-PSD of near-end and far-end
//
// In addition to updating the PSDs, also the filter diverge state is
// determined.
static void UpdateCoherenceSpectra(int mult,
                                   bool extended_filter_enabled,
                                   float efw[2][PART_LEN1],
                                   float dfw[2][PART_LEN1],
                                   float xfw[2][PART_LEN1],
                                   CoherenceState* coherence_state,
                                   short* filter_divergence_state,
                                   int* extreme_filter_divergence) {
  // Power estimate smoothing coefficients.
  const float* ptrGCoh =
      extended_filter_enabled
          ? WebRtcAec_kExtendedSmoothingCoefficients[mult - 1]
          : WebRtcAec_kNormalSmoothingCoefficients[mult - 1];
  int i;
  float sdSum = 0, seSum = 0;

  for (i = 0; i < PART_LEN1; i++) {
    coherence_state->sd[i] =
        ptrGCoh[0] * coherence_state->sd[i] +
        ptrGCoh[1] * (dfw[0][i] * dfw[0][i] + dfw[1][i] * dfw[1][i]);
    coherence_state->se[i] =
        ptrGCoh[0] * coherence_state->se[i] +
        ptrGCoh[1] * (efw[0][i] * efw[0][i] + efw[1][i] * efw[1][i]);
    // We threshold here to protect against the ill-effects of a zero farend.
    // The threshold is not arbitrarily chosen, but balances protection and
    // adverse interaction with the algorithm's tuning.
    // TODO(bjornv): investigate further why this is so sensitive.
    coherence_state->sx[i] =
        ptrGCoh[0] * coherence_state->sx[i] +
        ptrGCoh[1] *
            WEBRTC_SPL_MAX(xfw[0][i] * xfw[0][i] + xfw[1][i] * xfw[1][i],
                           WebRtcAec_kMinFarendPSD);

    coherence_state->sde[i][0] =
        ptrGCoh[0] * coherence_state->sde[i][0] +
        ptrGCoh[1] * (dfw[0][i] * efw[0][i] + dfw[1][i] * efw[1][i]);
    coherence_state->sde[i][1] =
        ptrGCoh[0] * coherence_state->sde[i][1] +
        ptrGCoh[1] * (dfw[0][i] * efw[1][i] - dfw[1][i] * efw[0][i]);

    coherence_state->sxd[i][0] =
        ptrGCoh[0] * coherence_state->sxd[i][0] +
        ptrGCoh[1] * (dfw[0][i] * xfw[0][i] + dfw[1][i] * xfw[1][i]);
    coherence_state->sxd[i][1] =
        ptrGCoh[0] * coherence_state->sxd[i][1] +
        ptrGCoh[1] * (dfw[0][i] * xfw[1][i] - dfw[1][i] * xfw[0][i]);

    sdSum += coherence_state->sd[i];
    seSum += coherence_state->se[i];
  }

  // Divergent filter safeguard update.
  *filter_divergence_state =
      (*filter_divergence_state ? 1.05f : 1.0f) * seSum > sdSum;

  // Signal extreme filter divergence if the error is significantly larger
  // than the nearend (13 dB).
  *extreme_filter_divergence = (seSum > (19.95f * sdSum));
}

// Window time domain data to be used by the fft.
__inline static void WindowData(float* x_windowed, const float* x) {
  int i;
  for (i = 0; i < PART_LEN; i++) {
    x_windowed[i] = x[i] * WebRtcAec_sqrtHanning[i];
    x_windowed[PART_LEN + i] =
        x[PART_LEN + i] * WebRtcAec_sqrtHanning[PART_LEN - i];
  }
}

// Puts fft output data into a complex valued array.
__inline static void StoreAsComplex(const float* data,
                                    float data_complex[2][PART_LEN1]) {
  int i;
  data_complex[0][0] = data[0];
  data_complex[1][0] = 0;
  for (i = 1; i < PART_LEN; i++) {
    data_complex[0][i] = data[2 * i];
    data_complex[1][i] = data[2 * i + 1];
  }
  data_complex[0][PART_LEN] = data[1];
  data_complex[1][PART_LEN] = 0;
}

static void ComputeCoherence(const CoherenceState* coherence_state,
                             float* cohde,
                             float* cohxd) {
  // Subband coherence
  for (int i = 0; i < PART_LEN1; i++) {
    cohde[i] = (coherence_state->sde[i][0] * coherence_state->sde[i][0] +
                coherence_state->sde[i][1] * coherence_state->sde[i][1]) /
               (coherence_state->sd[i] * coherence_state->se[i] + 1e-10f);
    cohxd[i] = (coherence_state->sxd[i][0] * coherence_state->sxd[i][0] +
                coherence_state->sxd[i][1] * coherence_state->sxd[i][1]) /
               (coherence_state->sx[i] * coherence_state->sd[i] + 1e-10f);
  }
}

static void GetHighbandGain(const float* lambda, float* nlpGainHband) {
  int i;

  *nlpGainHband = 0.0f;
  for (i = freqAvgIc; i < PART_LEN1 - 1; i++) {
    *nlpGainHband += lambda[i];
  }
  *nlpGainHband /= static_cast<float>(PART_LEN1 - 1 - freqAvgIc);
}

static void GenerateComplexNoise(uint32_t* seed, float noise[2][PART_LEN1]) {
  const float kPi2 = 6.28318530717959f;
  int16_t randW16[PART_LEN];
  WebRtcSpl_RandUArray(randW16, PART_LEN, seed);

  noise[0][0] = 0;
  noise[1][0] = 0;
  for (size_t i = 1; i < PART_LEN1; i++) {
    float tmp = kPi2 * randW16[i - 1] / 32768.f;
    noise[0][i] = cosf(tmp);
    noise[1][i] = -sinf(tmp);
  }
  noise[1][PART_LEN] = 0;
}

static void ComfortNoise(bool generate_high_frequency_noise,
                         uint32_t* seed,
                         float e_fft[2][PART_LEN1],
                         float high_frequency_comfort_noise[2][PART_LEN1],
                         const float* noise_spectrum,
                         const float* suppressor_gain) {
  float complex_noise[2][PART_LEN1];

  GenerateComplexNoise(seed, complex_noise);

  // Shape, scale and add comfort noise.
  for (int i = 1; i < PART_LEN1; ++i) {
    float noise_scaling =
        sqrtf(WEBRTC_SPL_MAX(1 - suppressor_gain[i] * suppressor_gain[i], 0)) *
        sqrtf(noise_spectrum[i]);
    e_fft[0][i] += noise_scaling * complex_noise[0][i];
    e_fft[1][i] += noise_scaling * complex_noise[1][i];
  }

  // Form comfort noise for higher frequencies.
  if (generate_high_frequency_noise) {
    // Compute average noise power and nlp gain over the second half of freq
    // spectrum (i.e., 4->8khz).
    int start_avg_band = PART_LEN1 / 2;
    float upper_bands_noise_power = 0.f;
    float upper_bands_suppressor_gain = 0.f;
    for (int i = start_avg_band; i < PART_LEN1; ++i) {
      upper_bands_noise_power += sqrtf(noise_spectrum[i]);
      upper_bands_suppressor_gain +=
          sqrtf(WEBRTC_SPL_MAX(1 - suppressor_gain[i] * suppressor_gain[i], 0));
    }
    upper_bands_noise_power /= (PART_LEN1 - start_avg_band);
    upper_bands_suppressor_gain /= (PART_LEN1 - start_avg_band);

    // Shape, scale and add comfort noise.
    float noise_scaling = upper_bands_suppressor_gain * upper_bands_noise_power;
    high_frequency_comfort_noise[0][0] = 0;
    high_frequency_comfort_noise[1][0] = 0;
    for (int i = 1; i < PART_LEN1; ++i) {
      high_frequency_comfort_noise[0][i] = noise_scaling * complex_noise[0][i];
      high_frequency_comfort_noise[1][i] = noise_scaling * complex_noise[1][i];
    }
    high_frequency_comfort_noise[1][PART_LEN] = 0;
  } else {
    memset(high_frequency_comfort_noise, 0,
           2 * PART_LEN1 * sizeof(high_frequency_comfort_noise[0][0]));
  }
}

static void InitLevel(PowerLevel* level) {
  const float kBigFloat = 1E17f;
  level->averagelevel.Reset();
  level->framelevel.Reset();
  level->minlevel = kBigFloat;
}

static void InitStats(Stats* stats) {
  stats->instant = kOffsetLevel;
  stats->average = kOffsetLevel;
  stats->max = kOffsetLevel;
  stats->min = kOffsetLevel * (-1);
  stats->sum = 0;
  stats->hisum = 0;
  stats->himean = kOffsetLevel;
  stats->counter = 0;
  stats->hicounter = 0;
}

static void InitMetrics(AecCore* self) {
  self->stateCounter = 0;
  InitLevel(&self->farlevel);
  InitLevel(&self->nearlevel);
  InitLevel(&self->linoutlevel);
  InitLevel(&self->nlpoutlevel);

  InitStats(&self->erl);
  InitStats(&self->erle);
  InitStats(&self->aNlp);
  InitStats(&self->rerl);

  self->divergent_filter_fraction.Reset();
}

static float CalculatePower(const float* in, size_t num_samples) {
  size_t k;
  float energy = 0.0f;

  for (k = 0; k < num_samples; ++k) {
    energy += in[k] * in[k];
  }
  return energy / num_samples;
}

static void UpdateLevel(PowerLevel* level, float power) {
  level->framelevel.AddValue(power);
  if (level->framelevel.EndOfBlock()) {
    const float new_frame_level = level->framelevel.GetLatestMean();
    if (new_frame_level > 0) {
      if (new_frame_level < level->minlevel) {
        level->minlevel = new_frame_level;  // New minimum.
      } else {
        level->minlevel *= (1 + 0.001f);  // Small increase.
      }
    }
    level->averagelevel.AddValue(new_frame_level);
  }
}

static void UpdateMetrics(AecCore* aec) {
  const float actThresholdNoisy = 8.0f;
  const float actThresholdClean = 40.0f;

  const float noisyPower = 300000.0f;

  float actThreshold;

  if (aec->echoState) {  // Check if echo is likely present
    aec->stateCounter++;
  }

  if (aec->linoutlevel.framelevel.EndOfBlock()) {
    aec->divergent_filter_fraction.AddObservation(
        aec->nearlevel, aec->linoutlevel, aec->nlpoutlevel);
  }

  if (aec->farlevel.averagelevel.EndOfBlock()) {
    if (aec->farlevel.minlevel < noisyPower) {
      actThreshold = actThresholdClean;
    } else {
      actThreshold = actThresholdNoisy;
    }

    const float far_average_level = aec->farlevel.averagelevel.GetLatestMean();

    // The last condition is to let estimation be made in active far-end
    // segments only.
    if ((aec->stateCounter > (0.5f * kCountLen * kSubCountLen)) &&
        (aec->farlevel.framelevel.EndOfBlock()) &&
        (far_average_level > (actThreshold * aec->farlevel.minlevel))) {
      // ERL: error return loss.
      const float near_average_level =
          aec->nearlevel.averagelevel.GetLatestMean();
      UpdateLogRatioMetric(&aec->erl, far_average_level, near_average_level);

      // A_NLP: error return loss enhanced before the nonlinear suppression.
      const float linout_average_level =
          aec->linoutlevel.averagelevel.GetLatestMean();
      UpdateLogRatioMetric(&aec->aNlp, near_average_level,
                           linout_average_level);

      // ERLE: error return loss enhanced.
      const float nlpout_average_level =
          aec->nlpoutlevel.averagelevel.GetLatestMean();
      UpdateLogRatioMetric(&aec->erle, near_average_level,
                           nlpout_average_level);
    }

    aec->stateCounter = 0;
  }
}

static void UpdateDelayMetrics(AecCore* self) {
  int i = 0;
  int delay_values = 0;
  int median = 0;
  int lookahead = WebRtc_lookahead(self->delay_estimator);
  const int kMsPerBlock = PART_LEN / (self->mult * 8);
  int64_t l1_norm = 0;

  if (self->num_delay_values == 0) {
    // We have no new delay value data. Even though -1 is a valid |median| in
    // the sense that we allow negative values, it will practically never be
    // used since multiples of |kMsPerBlock| will always be returned.
    // We therefore use -1 to indicate in the logs that the delay estimator was
    // not able to estimate the delay.
    self->delay_median = -1;
    self->delay_std = -1;
    self->fraction_poor_delays = -1;
    return;
  }

  // Start value for median count down.
  delay_values = self->num_delay_values >> 1;
  // Get median of delay values since last update.
  for (i = 0; i < kHistorySizeBlocks; i++) {
    delay_values -= self->delay_histogram[i];
    if (delay_values < 0) {
      median = i;
      break;
    }
  }
  // Account for lookahead.
  self->delay_median = (median - lookahead) * kMsPerBlock;

  // Calculate the L1 norm, with median value as central moment.
  for (i = 0; i < kHistorySizeBlocks; i++) {
    l1_norm += abs(i - median) * self->delay_histogram[i];
  }
  self->delay_std = static_cast<int>((l1_norm + self->num_delay_values / 2) /
                                     self->num_delay_values) *
                    kMsPerBlock;

  // Determine fraction of delays that are out of bounds, that is, either
  // negative (anti-causal system) or larger than the AEC filter length.
  {
    int num_delays_out_of_bounds = self->num_delay_values;
    const int histogram_length =
        sizeof(self->delay_histogram) / sizeof(self->delay_histogram[0]);
    for (i = lookahead; i < lookahead + self->num_partitions; ++i) {
      if (i < histogram_length)
        num_delays_out_of_bounds -= self->delay_histogram[i];
    }
    self->fraction_poor_delays =
        static_cast<float>(num_delays_out_of_bounds) / self->num_delay_values;
  }

  // Reset histogram.
  memset(self->delay_histogram, 0, sizeof(self->delay_histogram));
  self->num_delay_values = 0;
}

static void ScaledInverseFft(const OouraFft& ooura_fft,
                             float freq_data[2][PART_LEN1],
                             float time_data[PART_LEN2],
                             float scale,
                             int conjugate) {
  int i;
  const float normalization = scale / static_cast<float>(PART_LEN2);
  const float sign = (conjugate ? -1 : 1);
  time_data[0] = freq_data[0][0] * normalization;
  time_data[1] = freq_data[0][PART_LEN] * normalization;
  for (i = 1; i < PART_LEN; i++) {
    time_data[2 * i] = freq_data[0][i] * normalization;
    time_data[2 * i + 1] = sign * freq_data[1][i] * normalization;
  }
  ooura_fft.InverseFft(time_data);
}

static void Fft(const OouraFft& ooura_fft,
                float time_data[PART_LEN2],
                float freq_data[2][PART_LEN1]) {
  int i;
  ooura_fft.Fft(time_data);

  // Reorder fft output data.
  freq_data[1][0] = 0;
  freq_data[1][PART_LEN] = 0;
  freq_data[0][0] = time_data[0];
  freq_data[0][PART_LEN] = time_data[1];
  for (i = 1; i < PART_LEN; i++) {
    freq_data[0][i] = time_data[2 * i];
    freq_data[1][i] = time_data[2 * i + 1];
  }
}

static int SignalBasedDelayCorrection(AecCore* self) {
  int delay_correction = 0;
  int last_delay = -2;
  RTC_DCHECK(self);
#if !defined(WEBRTC_ANDROID)
  // On desktops, turn on correction after |kDelayCorrectionStart| frames.  This
  // is to let the delay estimation get a chance to converge.  Also, if the
  // playout audio volume is low (or even muted) the delay estimation can return
  // a very large delay, which will break the AEC if it is applied.
  if (self->frame_count < kDelayCorrectionStart) {
    self->data_dumper->DumpRaw("aec_da_reported_delay", 1, &last_delay);
    return 0;
  }
#endif

  // 1. Check for non-negative delay estimate.  Note that the estimates we get
  //    from the delay estimation are not compensated for lookahead.  Hence, a
  //    negative |last_delay| is an invalid one.
  // 2. Verify that there is a delay change.  In addition, only allow a change
  //    if the delay is outside a certain region taking the AEC filter length
  //    into account.
  // TODO(bjornv): Investigate if we can remove the non-zero delay change check.
  // 3. Only allow delay correction if the delay estimation quality exceeds
  //    |delay_quality_threshold|.
  // 4. Finally, verify that the proposed |delay_correction| is feasible by
  //    comparing with the size of the far-end buffer.
  last_delay = WebRtc_last_delay(self->delay_estimator);
  self->data_dumper->DumpRaw("aec_da_reported_delay", 1, &last_delay);
  if ((last_delay >= 0) && (last_delay != self->previous_delay) &&
      (WebRtc_last_delay_quality(self->delay_estimator) >
       self->delay_quality_threshold)) {
    int delay = last_delay - WebRtc_lookahead(self->delay_estimator);
    // Allow for a slack in the actual delay, defined by a |lower_bound| and an
    // |upper_bound|.  The adaptive echo cancellation filter is currently
    // |num_partitions| (of 64 samples) long.  If the delay estimate is negative
    // or at least 3/4 of the filter length we open up for correction.
    const int lower_bound = 0;
    const int upper_bound = self->num_partitions * 3 / 4;
    const int do_correction = delay <= lower_bound || delay > upper_bound;
    if (do_correction == 1) {
      int available_read = self->farend_block_buffer_.Size();
      // With |shift_offset| we gradually rely on the delay estimates.  For
      // positive delays we reduce the correction by |shift_offset| to lower the
      // risk of pushing the AEC into a non causal state.  For negative delays
      // we rely on the values up to a rounding error, hence compensate by 1
      // element to make sure to push the delay into the causal region.
      delay_correction = -delay;
      delay_correction += delay > self->shift_offset ? self->shift_offset : 1;
      self->shift_offset--;
      self->shift_offset = (self->shift_offset <= 1 ? 1 : self->shift_offset);
      if (delay_correction > available_read - self->mult - 1) {
        // There is not enough data in the buffer to perform this shift.  Hence,
        // we do not rely on the delay estimate and do nothing.
        delay_correction = 0;
      } else {
        self->previous_delay = last_delay;
        ++self->delay_correction_count;
      }
    }
  }
  // Update the |delay_quality_threshold| once we have our first delay
  // correction.
  if (self->delay_correction_count > 0) {
    float delay_quality = WebRtc_last_delay_quality(self->delay_estimator);
    delay_quality =
        (delay_quality > kDelayQualityThresholdMax ? kDelayQualityThresholdMax
                                                   : delay_quality);
    self->delay_quality_threshold =
        (delay_quality > self->delay_quality_threshold
             ? delay_quality
             : self->delay_quality_threshold);
  }
  self->data_dumper->DumpRaw("aec_da_delay_correction", 1, &delay_correction);

  return delay_correction;
}

static void RegressorPower(
    int num_partitions,
    int latest_added_partition,
    float x_fft_buf[2][kExtendedNumPartitions * PART_LEN1],
    float x_pow[PART_LEN1]) {
  RTC_DCHECK_LT(latest_added_partition, num_partitions);
  memset(x_pow, 0, PART_LEN1 * sizeof(x_pow[0]));

  int partition = latest_added_partition;
  int x_fft_buf_position = partition * PART_LEN1;
  for (int i = 0; i < num_partitions; ++i) {
    for (int bin = 0; bin < PART_LEN1; ++bin) {
      float re = x_fft_buf[0][x_fft_buf_position];
      float im = x_fft_buf[1][x_fft_buf_position];
      x_pow[bin] += re * re + im * im;
      ++x_fft_buf_position;
    }

    ++partition;
    if (partition == num_partitions) {
      partition = 0;
      RTC_DCHECK_EQ(num_partitions * PART_LEN1, x_fft_buf_position);
      x_fft_buf_position = 0;
    }
  }
}
#ifndef AUMDF_FILTER
static void EchoSubtraction(
    const OouraFft& ooura_fft,
    int num_partitions,
    int extended_filter_enabled,
    int* extreme_filter_divergence,
    float filter_step_size,
    float error_threshold,
    float* x_fft,
    int* x_fft_buf_block_pos,
    float x_fft_buf[2][kExtendedNumPartitions * PART_LEN1],
    float* const y,
    float x_pow[PART_LEN1],
    float h_fft_buf[2][kExtendedNumPartitions * PART_LEN1],
    float echo_subtractor_output[PART_LEN]) {
  float s_fft[2][PART_LEN1];
  float e_extended[PART_LEN2];
  float s_extended[PART_LEN2];
  float* s;
  float e[PART_LEN];
  float e_fft[2][PART_LEN1];
  int i;

  // Update the x_fft_buf block position.
  (*x_fft_buf_block_pos)--;
  if ((*x_fft_buf_block_pos) == -1) {
    *x_fft_buf_block_pos = num_partitions - 1;
  }

  // Buffer x_fft.
  memcpy(x_fft_buf[0] + (*x_fft_buf_block_pos) * PART_LEN1, x_fft,
         sizeof(float) * PART_LEN1);
  memcpy(x_fft_buf[1] + (*x_fft_buf_block_pos) * PART_LEN1, &x_fft[PART_LEN1],
         sizeof(float) * PART_LEN1);

  memset(s_fft, 0, sizeof(s_fft));

  // Conditionally reset the echo subtraction filter if the filter has diverged
  // significantly.
  if (!extended_filter_enabled && *extreme_filter_divergence) {
    memset(h_fft_buf, 0,
           2 * kExtendedNumPartitions * PART_LEN1 * sizeof(h_fft_buf[0][0]));
    *extreme_filter_divergence = 0;
  }

  // Produce echo estimate s_fft.
  WebRtcAec_FilterFar(num_partitions, *x_fft_buf_block_pos, x_fft_buf,
                      h_fft_buf, s_fft);

  // Compute the time-domain echo estimate s.
  ScaledInverseFft(ooura_fft, s_fft, s_extended, 2.0f, 0);
  s = &s_extended[PART_LEN];

  // Compute the time-domain echo prediction error.
  for (i = 0; i < PART_LEN; ++i) {
    e[i] = y[i] - s[i];
  }

  // Compute the frequency domain echo prediction error.
  memset(e_extended, 0, sizeof(float) * PART_LEN);
  memcpy(e_extended + PART_LEN, e, sizeof(float) * PART_LEN);
  Fft(ooura_fft, e_extended, e_fft);

  // Scale error signal inversely with far power.
  WebRtcAec_ScaleErrorSignal(filter_step_size, error_threshold, x_pow, e_fft);
  WebRtcAec_FilterAdaptation(ooura_fft, num_partitions, *x_fft_buf_block_pos,
                             x_fft_buf, e_fft, h_fft_buf);
  memcpy(echo_subtractor_output, e, sizeof(float) * PART_LEN);
}
#endif
static void FormSuppressionGain(AecCore* aec,
                                float cohde[PART_LEN1],
                                float cohxd[PART_LEN1],
                                float hNl[PART_LEN1]) {
  float hNlDeAvg, hNlXdAvg;
  float hNlPref[kPrefBandSize];
  float hNlFb = 0, hNlFbLow = 0;
  const int prefBandSize = kPrefBandSize / aec->mult;
  const float prefBandQuant = 0.75f, prefBandQuantLow = 0.5f;
  const int minPrefBand = 4 / aec->mult;
  // Power estimate smoothing coefficients.
  const float* min_overdrive = aec->extended_filter_enabled
                                   ? kExtendedMinOverDrive
                                   : kNormalMinOverDrive;

  hNlXdAvg = 0;
  for (int i = minPrefBand; i < prefBandSize + minPrefBand; ++i) {
    hNlXdAvg += cohxd[i];
  }
  hNlXdAvg /= prefBandSize;
  hNlXdAvg = 1 - hNlXdAvg;

  hNlDeAvg = 0;
  for (int i = minPrefBand; i < prefBandSize + minPrefBand; ++i) {
    hNlDeAvg += cohde[i];
  }
  hNlDeAvg /= prefBandSize;

  if (hNlXdAvg < 0.75f && hNlXdAvg < aec->hNlXdAvgMin) {
    aec->hNlXdAvgMin = hNlXdAvg;
  }

  if (hNlDeAvg > 0.98f && hNlXdAvg > 0.9f) {
    aec->stNearState = 1;
  } else if (hNlDeAvg < 0.95f || hNlXdAvg < 0.8f) {
    aec->stNearState = 0;
  }

  if (aec->hNlXdAvgMin == 1) {
    aec->echoState = 0;
    aec->overDrive = min_overdrive[aec->nlp_mode];

    if (aec->stNearState == 1) {
      memcpy(hNl, cohde, sizeof(hNl[0]) * PART_LEN1);
      hNlFb = hNlDeAvg;
      hNlFbLow = hNlDeAvg;
    } else {
      for (int i = 0; i < PART_LEN1; ++i) {
        hNl[i] = 1 - cohxd[i];
        hNl[i] = std::max(hNl[i], 0.f);
      }
      hNlFb = hNlXdAvg;
      hNlFbLow = hNlXdAvg;
    }
  } else {
    if (aec->stNearState == 1) {
      aec->echoState = 0;
      memcpy(hNl, cohde, sizeof(hNl[0]) * PART_LEN1);
      hNlFb = hNlDeAvg;
      hNlFbLow = hNlDeAvg;
    } else {
      aec->echoState = 1;
      for (int i = 0; i < PART_LEN1; ++i) {
        hNl[i] = WEBRTC_SPL_MIN(cohde[i], 1 - cohxd[i]);
        hNl[i] = std::max(hNl[i], 0.f);
      }

      // Select an order statistic from the preferred bands.
      // TODO(peah): Using quicksort now, but a selection algorithm may be
      // preferred.
      memcpy(hNlPref, &hNl[minPrefBand], sizeof(float) * prefBandSize);
      qsort(hNlPref, prefBandSize, sizeof(float), CmpFloat);
      hNlFb =
          hNlPref[static_cast<int>(floor(prefBandQuant * (prefBandSize - 1)))];
      hNlFbLow = hNlPref[static_cast<int>(
          floor(prefBandQuantLow * (prefBandSize - 1)))];
    }
  }
#ifdef AUMDF_FILTER
  for (int i = 0; i < PART_LEN1; ++i) {
    hNl[i] = WEBRTC_SPL_MIN(cohde[i], 1 - cohxd[i]);
  }
#endif

  // Track the local filter minimum to determine suppression overdrive.
  if (hNlFbLow < 0.6f && hNlFbLow < aec->hNlFbLocalMin) {
    aec->hNlFbLocalMin = hNlFbLow;
    aec->hNlFbMin = hNlFbLow;
    aec->hNlNewMin = 1;
    aec->hNlMinCtr = 0;
  }
  aec->hNlFbLocalMin =
      WEBRTC_SPL_MIN(aec->hNlFbLocalMin + 0.0008f / aec->mult, 1);
  aec->hNlXdAvgMin = WEBRTC_SPL_MIN(aec->hNlXdAvgMin + 0.0006f / aec->mult, 1);

  if (aec->hNlNewMin == 1) {
    aec->hNlMinCtr++;
  }
  if (aec->hNlMinCtr == 2) {
    aec->hNlNewMin = 0;
    aec->hNlMinCtr = 0;
    aec->overDrive = WEBRTC_SPL_MAX(
        kTargetSupp[aec->nlp_mode] /
            static_cast<float>(log(aec->hNlFbMin + 1e-10f) + 1e-10f),
        min_overdrive[aec->nlp_mode]);
  }

  // Smooth the overdrive.
  if (aec->overDrive < aec->overdrive_scaling) {
    aec->overdrive_scaling =
        0.99f * aec->overdrive_scaling + 0.01f * aec->overDrive;
  } else {
    aec->overdrive_scaling =
        0.9f * aec->overdrive_scaling + 0.1f * aec->overDrive;
  }

  if (aec->nlp_mode == 3)
  {
	  aec->overdrive_scaling = 1.0;
  }

  // Apply the overdrive.
  WebRtcAec_Overdrive(aec->overdrive_scaling, hNlFb, hNl);
}

#ifdef AUMDF_FILTER
static int PartitionDelayMdf(int num_partitions,
                             float h_fft_buf[PART_LEN * PART_LEN2]) {
// Measures the energy in each filter partition and returns the partition with
// highest energy.
// TODO(bjornv): Spread computational cost by computing one partition per
// block?
#if 0
	float wfEnMax = 0;
	int i;
	int delay = 0;

	for (i = 0; i < num_partitions; i++)
	{
		int j;
		int pos = i * PART_LEN2;
		float wfEn = 0;
		for (j = 0; j < PART_LEN2; j++)
		{
			wfEn += h_fft_buf[pos + j] * h_fft_buf[pos + j] + h_fft_buf[pos + j + 1] * h_fft_buf[pos + j + 1];
		}

		if (wfEn > wfEnMax)
		{
			wfEnMax = wfEn;
			delay = i;
		}
	}
	printf("delay = %d\n", delay);
	return delay;
#else
  float wfEnMax = 0;
  int i;
  int delay = 0;
  float acc[MM] = {0.0};
  int N = 128;

  int j;
  for (i = 0; i < num_partitions; i++) {
    acc[i] = 0;
  }
  for (j = 0; j < num_partitions; j++) {
    for (i = 0; i < N - 1; i += 2) {
      acc[j] +=
          (h_fft_buf[i] * h_fft_buf[i] + h_fft_buf[i + 1] * h_fft_buf[i + 1]);
    }
    h_fft_buf += N;
  }

  wfEnMax = acc[0];
  for (i = 1; i < num_partitions; i++) {
    if (acc[i] > wfEnMax) {
      wfEnMax = acc[i];
      delay = i;
    }
  }
  // printf("delay = %d\n", delay);
  return delay;
#endif
}
#endif

static void EchoSuppression(const OouraFft& ooura_fft,
                            AecCore* aec,
                            float leakestimate,
                            float* nearend_extended_block_lowest_band,
                            float farend_extended_block[PART_LEN2],
                            float* echo_subtractor_output,
                            float output[NUM_HIGH_BANDS_MAX + 1][PART_LEN]) {
  float efw[2][PART_LEN1];
  float xfw[2][PART_LEN1];
  float dfw[2][PART_LEN1];
  float comfortNoiseHband[2][PART_LEN1];
  float fft[PART_LEN2];
  float nlpGainHband;
  int i;
  size_t j;

  // Coherence and non-linear filter
  float cohde[PART_LEN1], cohxd[PART_LEN1];
  float hNl[PART_LEN1];

  // Filter energy
  const int delayEstInterval = 10 * aec->mult;

  float* xfw_ptr = NULL;

  // Update eBuf with echo subtractor output.
  memcpy(aec->eBuf + PART_LEN, echo_subtractor_output,
         sizeof(float) * PART_LEN);

  // Analysis filter banks for the echo suppressor.
  // Windowed near-end ffts.
  WindowData(fft, nearend_extended_block_lowest_band);
  ooura_fft.Fft(fft);
  StoreAsComplex(fft, dfw);

  // Windowed echo suppressor output ffts.
  WindowData(fft, aec->eBuf);
  ooura_fft.Fft(fft);
  StoreAsComplex(fft, efw);

  // NLP

  // Convert far-end partition to the frequency domain with windowing.
  WindowData(fft, farend_extended_block);
  Fft(ooura_fft, fft, xfw);
  xfw_ptr = &xfw[0][0];

  // Buffer far.
  memcpy(aec->xfwBuf, xfw_ptr, sizeof(float) * 2 * PART_LEN1);

  aec->delayEstCtr++;
  if (aec->delayEstCtr == delayEstInterval) {
    aec->delayEstCtr = 0;
#ifndef AUMDF_FILTER
    aec->delayIdx = WebRtcAec_PartitionDelay(aec->num_partitions, aec->wfBuf);
#else
    aec->delayIdx = PartitionDelayMdf(aec->num_partitions, aec->st->W);
#endif
  }

  aec->data_dumper->DumpRaw("aec_nlp_delay", 1, &aec->delayIdx);

  // Use delayed far.
  memcpy(xfw, aec->xfwBuf + aec->delayIdx * PART_LEN1,
         sizeof(xfw[0][0]) * 2 * PART_LEN1);

  WebRtcAec_UpdateCoherenceSpectra(aec->mult, aec->extended_filter_enabled == 1,
                                   efw, dfw, xfw, &aec->coherence_state,
                                   &aec->divergeState,
                                   &aec->extreme_filter_divergence);

  WebRtcAec_ComputeCoherence(&aec->coherence_state, cohde, cohxd);

  // Select the microphone signal as output if the filter is deemed to have
  // diverged.
  if (aec->divergeState) {
    memcpy(efw, dfw, sizeof(efw[0][0]) * 2 * PART_LEN1);
  }

  FormSuppressionGain(aec, cohde, cohxd, hNl);

  aec->data_dumper->DumpRaw("aec_nlp_gain", PART_LEN1, hNl);

  if (aec->nlp_mode != 3)
  {
#ifdef AUMDF_FILTER
  float SubbandTotalSum = 0.0;
  float SubbandAverage = 0.0;
  for (i = 0; i < 8; i++) {
    SubbandTotalSum = SubbandTotalSum + hNl[i];
  }
  SubbandAverage = SubbandTotalSum / 8;
  if (SubbandAverage < 0.125) {
    for (i = 0; i < PART_LEN1; i++) {
      hNl[i] = 0.0;
    }
  } else if (SubbandAverage >= 0.125 && SubbandAverage < 0.25) {
    for (i = 0; i < PART_LEN1; i++) {
      hNl[i] = powf(hNl[i], 3);
    }
  } else if (SubbandAverage >= 0.25 && SubbandAverage < 0.5) {
    for (i = 0; i < PART_LEN1; i++) {
      hNl[i] = powf(hNl[i], 1.5);
    }
  }
#endif

  if (leakestimate >= 0.5) {
    for (i = 0; i < PART_LEN1; i++) {
      hNl[i] = powf(hNl[i], 3);
    }
  }
}

  WebRtcAec_Suppress(hNl, efw);

  // Add comfort noise.
  for (i = 0; i < PART_LEN1; i++) {
    aec->noisePow[i] = 0.0;
  }
  ComfortNoise(aec->num_bands > 1, &aec->seed, efw, comfortNoiseHband,
               aec->noisePow, hNl);

  // Inverse error fft.
  ScaledInverseFft(ooura_fft, efw, fft, 2.0f, 1);

  // Overlap and add to obtain output.
  for (i = 0; i < PART_LEN; i++) {
    output[0][i] = (fft[i] * WebRtcAec_sqrtHanning[i] +
                    aec->outBuf[i] * WebRtcAec_sqrtHanning[PART_LEN - i]);

    // Saturate output to keep it in the allowed range.
    output[0][i] = WEBRTC_SPL_SAT(WEBRTC_SPL_WORD16_MAX, output[0][i],
                                  WEBRTC_SPL_WORD16_MIN);
  }
  memcpy(aec->outBuf, &fft[PART_LEN], PART_LEN * sizeof(aec->outBuf[0]));

  // For H band
  if (aec->num_bands > 1) {
    // H band gain
    // average nlp over low band: average over second half of freq spectrum
    // (4->8khz)
    GetHighbandGain(hNl, &nlpGainHband);

    // Inverse comfort_noise
    ScaledInverseFft(ooura_fft, comfortNoiseHband, fft, 2.0f, 0);

    // compute gain factor
    for (j = 1; j < aec->num_bands; ++j) {
      for (i = 0; i < PART_LEN; i++) {
        output[j][i] = aec->previous_nearend_block[j][i] * nlpGainHband;
      }
    }

    // Add some comfort noise where Hband is attenuated.
    for (i = 0; i < PART_LEN; i++) {
      output[1][i] += cnScaleHband * fft[i];
    }

    // Saturate output to keep it in the allowed range.
    for (j = 1; j < aec->num_bands; ++j) {
      for (i = 0; i < PART_LEN; i++) {
        output[j][i] = WEBRTC_SPL_SAT(WEBRTC_SPL_WORD16_MAX, output[j][i],
                                      WEBRTC_SPL_WORD16_MIN);
      }
    }
  }

  // Copy the current block to the old position.
  memcpy(aec->eBuf, aec->eBuf + PART_LEN, sizeof(float) * PART_LEN);

  memmove(aec->xfwBuf + PART_LEN1, aec->xfwBuf,
          sizeof(aec->xfwBuf) - sizeof(complex_t) * PART_LEN1);
}

#ifdef AUMDF_FILTER
static void kf_bfly2(kiss_fft_cpx* Fout,
                     const size_t fstride,
                     const kiss_fft_cfg st,
                     int m,
                     int N,
                     int mm) {
  kiss_fft_cpx* Fout2;
  kiss_fft_cpx* tw1;
  kiss_fft_cpx t;
  if (!st->inverse) {
    int i, j;
    kiss_fft_cpx* Fout_beg = Fout;
    for (i = 0; i < N; i++) {
      Fout = Fout_beg + i * mm;
      Fout2 = Fout + m;
      tw1 = st->twiddles;
      for (j = 0; j < m; j++) {
        /* Almost the same as the code path below, except that we divide the
        input by two (while keeping the best accuracy possible) */
        float tr, ti;
        tr = SHR32(
            SUB32(MULT16_16(Fout2->r, tw1->r), MULT16_16(Fout2->i, tw1->i)), 1);
        ti = SHR32(
            ADD32(MULT16_16(Fout2->i, tw1->r), MULT16_16(Fout2->r, tw1->i)), 1);
        tw1 += fstride;
        Fout2->r = PSHR32(SUB32(SHL32(EXTEND32(Fout->r), 14), tr), 15);
        Fout2->i = PSHR32(SUB32(SHL32(EXTEND32(Fout->i), 14), ti), 15);
        Fout->r = PSHR32(ADD32(SHL32(EXTEND32(Fout->r), 14), tr), 15);
        Fout->i = PSHR32(ADD32(SHL32(EXTEND32(Fout->i), 14), ti), 15);
        ++Fout2;
        ++Fout;
      }
    }
  } else {
    int i, j;
    kiss_fft_cpx* Fout_beg = Fout;
    for (i = 0; i < N; i++) {
      Fout = Fout_beg + i * mm;
      Fout2 = Fout + m;
      tw1 = st->twiddles;
      for (j = 0; j < m; j++) {
        C_MUL(t, *Fout2, *tw1);
        tw1 += fstride;
        C_SUB(*Fout2, *Fout, t);
        C_ADDTO(*Fout, t);
        ++Fout2;
        ++Fout;
      }
    }
  }
}

static void kf_bfly4(kiss_fft_cpx* Fout,
                     const size_t fstride,
                     const kiss_fft_cfg st,
                     int m,
                     int N,
                     int mm) {
  kiss_fft_cpx *tw1, *tw2, *tw3;
  kiss_fft_cpx scratch[6];
  const size_t m2 = 2 * m;
  const size_t m3 = 3 * m;
  int i, j;

  if (st->inverse) {
    kiss_fft_cpx* Fout_beg = Fout;
    for (i = 0; i < N; i++) {
      Fout = Fout_beg + i * mm;
      tw3 = tw2 = tw1 = st->twiddles;
      for (j = 0; j < m; j++) {
        C_MUL(scratch[0], Fout[m], *tw1);
        C_MUL(scratch[1], Fout[m2], *tw2);
        C_MUL(scratch[2], Fout[m3], *tw3);

        C_SUB(scratch[5], *Fout, scratch[1]);
        C_ADDTO(*Fout, scratch[1]);
        C_ADD(scratch[3], scratch[0], scratch[2]);
        C_SUB(scratch[4], scratch[0], scratch[2]);
        C_SUB(Fout[m2], *Fout, scratch[3]);
        tw1 += fstride;
        tw2 += fstride * 2;
        tw3 += fstride * 3;
        C_ADDTO(*Fout, scratch[3]);

        Fout[m].r = scratch[5].r - scratch[4].i;
        Fout[m].i = scratch[5].i + scratch[4].r;
        Fout[m3].r = scratch[5].r + scratch[4].i;
        Fout[m3].i = scratch[5].i - scratch[4].r;
        ++Fout;
      }
    }
  } else {
    kiss_fft_cpx* Fout_beg = Fout;
    for (i = 0; i < N; i++) {
      Fout = Fout_beg + i * mm;
      tw3 = tw2 = tw1 = st->twiddles;
      for (j = 0; j < m; j++) {
        C_MUL4(scratch[0], Fout[m], *tw1);
        C_MUL4(scratch[1], Fout[m2], *tw2);
        C_MUL4(scratch[2], Fout[m3], *tw3);

        Fout->r = PSHR16(Fout->r, 2);
        Fout->i = PSHR16(Fout->i, 2);
        C_SUB(scratch[5], *Fout, scratch[1]);
        C_ADDTO(*Fout, scratch[1]);
        C_ADD(scratch[3], scratch[0], scratch[2]);
        C_SUB(scratch[4], scratch[0], scratch[2]);
        Fout[m2].r = PSHR16(Fout[m2].r, 2);
        Fout[m2].i = PSHR16(Fout[m2].i, 2);
        C_SUB(Fout[m2], *Fout, scratch[3]);
        tw1 += fstride;
        tw2 += fstride * 2;
        tw3 += fstride * 3;
        C_ADDTO(*Fout, scratch[3]);

        Fout[m].r = scratch[5].r + scratch[4].i;
        Fout[m].i = scratch[5].i - scratch[4].r;
        Fout[m3].r = scratch[5].r - scratch[4].i;
        Fout[m3].i = scratch[5].i + scratch[4].r;
        ++Fout;
      }
    }
  }
}

static void kf_bfly3(kiss_fft_cpx* Fout,
                     const size_t fstride,
                     const kiss_fft_cfg st,
                     size_t m) {
  size_t k = m;
  const size_t m2 = 2 * m;
  kiss_fft_cpx *tw1, *tw2;
  kiss_fft_cpx scratch[5];
  kiss_fft_cpx epi3;
  epi3 = st->twiddles[fstride * m];

  tw1 = tw2 = st->twiddles;

  do {
    if (!st->inverse) {
      C_FIXDIV(*Fout, 3);
      C_FIXDIV(Fout[m], 3);
      C_FIXDIV(Fout[m2], 3);
    }

    C_MUL(scratch[1], Fout[m], *tw1);
    C_MUL(scratch[2], Fout[m2], *tw2);

    C_ADD(scratch[3], scratch[1], scratch[2]);
    C_SUB(scratch[0], scratch[1], scratch[2]);
    tw1 += fstride;
    tw2 += fstride * 2;

    Fout[m].r = Fout->r - HALF_OF(scratch[3].r);
    Fout[m].i = Fout->i - HALF_OF(scratch[3].i);

    C_MULBYSCALAR(scratch[0], epi3.i);

    C_ADDTO(*Fout, scratch[3]);

    Fout[m2].r = Fout[m].r + scratch[0].i;
    Fout[m2].i = Fout[m].i - scratch[0].r;

    Fout[m].r -= scratch[0].i;
    Fout[m].i += scratch[0].r;

    ++Fout;
  } while (--k);
}

static void kf_bfly5(kiss_fft_cpx* Fout,
                     const size_t fstride,
                     const kiss_fft_cfg st,
                     int m) {
  kiss_fft_cpx *Fout0, *Fout1, *Fout2, *Fout3, *Fout4;
  int u;
  kiss_fft_cpx scratch[13];
  kiss_fft_cpx* twiddles = st->twiddles;
  kiss_fft_cpx* tw;
  kiss_fft_cpx ya, yb;
  ya = twiddles[fstride * m];
  yb = twiddles[fstride * 2 * m];

  Fout0 = Fout;
  Fout1 = Fout0 + m;
  Fout2 = Fout0 + 2 * m;
  Fout3 = Fout0 + 3 * m;
  Fout4 = Fout0 + 4 * m;

  tw = st->twiddles;
  for (u = 0; u < m; ++u) {
    if (!st->inverse) {
      C_FIXDIV(*Fout0, 5);
      C_FIXDIV(*Fout1, 5);
      C_FIXDIV(*Fout2, 5);
      C_FIXDIV(*Fout3, 5);
      C_FIXDIV(*Fout4, 5);
    }
    scratch[0] = *Fout0;

    C_MUL(scratch[1], *Fout1, tw[u * fstride]);
    C_MUL(scratch[2], *Fout2, tw[2 * u * fstride]);
    C_MUL(scratch[3], *Fout3, tw[3 * u * fstride]);
    C_MUL(scratch[4], *Fout4, tw[4 * u * fstride]);

    C_ADD(scratch[7], scratch[1], scratch[4]);
    C_SUB(scratch[10], scratch[1], scratch[4]);
    C_ADD(scratch[8], scratch[2], scratch[3]);
    C_SUB(scratch[9], scratch[2], scratch[3]);

    Fout0->r += scratch[7].r + scratch[8].r;
    Fout0->i += scratch[7].i + scratch[8].i;

    scratch[5].r =
        scratch[0].r + S_MUL(scratch[7].r, ya.r) + S_MUL(scratch[8].r, yb.r);
    scratch[5].i =
        scratch[0].i + S_MUL(scratch[7].i, ya.r) + S_MUL(scratch[8].i, yb.r);

    scratch[6].r = S_MUL(scratch[10].i, ya.i) + S_MUL(scratch[9].i, yb.i);
    scratch[6].i = -S_MUL(scratch[10].r, ya.i) - S_MUL(scratch[9].r, yb.i);

    C_SUB(*Fout1, scratch[5], scratch[6]);
    C_ADD(*Fout4, scratch[5], scratch[6]);

    scratch[11].r =
        scratch[0].r + S_MUL(scratch[7].r, yb.r) + S_MUL(scratch[8].r, ya.r);
    scratch[11].i =
        scratch[0].i + S_MUL(scratch[7].i, yb.r) + S_MUL(scratch[8].i, ya.r);
    scratch[12].r = -S_MUL(scratch[10].i, yb.i) + S_MUL(scratch[9].i, ya.i);
    scratch[12].i = S_MUL(scratch[10].r, yb.i) - S_MUL(scratch[9].r, ya.i);

    C_ADD(*Fout2, scratch[11], scratch[12]);
    C_SUB(*Fout3, scratch[11], scratch[12]);

    ++Fout0;
    ++Fout1;
    ++Fout2;
    ++Fout3;
    ++Fout4;
  }
}

/* perform the butterfly for one stage of a mixed radix FFT */
static void kf_bfly_generic(kiss_fft_cpx* Fout,
                            const size_t fstride,
                            const kiss_fft_cfg st,
                            int m,
                            int p) {
  int u, k, q1, q;
  kiss_fft_cpx* twiddles = st->twiddles;
  kiss_fft_cpx t;
  kiss_fft_cpx scratchbuf[17];
  int Norig = st->nfft;

  if (p > 17)
    // speex_fatal("KissFFT: max radix supported is 17");

    for (u = 0; u < m; ++u) {
      k = u;
      for (q1 = 0; q1 < p; ++q1) {
        scratchbuf[q1] = Fout[k];
        if (!st->inverse) {
          C_FIXDIV(scratchbuf[q1], p);
        }
        k += m;
      }

      k = u;
      for (q1 = 0; q1 < p; ++q1) {
        int twidx = 0;
        Fout[k] = scratchbuf[0];
        for (q = 1; q < p; ++q) {
          twidx += fstride * k;
          if (twidx >= Norig)
            twidx -= Norig;
          C_MUL(t, scratchbuf[q], twiddles[twidx]);
          C_ADDTO(Fout[k], t);
        }
        k += m;
      }
    }
}

static void kf_shuffle(kiss_fft_cpx* Fout,
                       const kiss_fft_cpx* f,
                       const size_t fstride,
                       int in_stride,
                       int* factors,
                       const kiss_fft_cfg st) {
  const int p = *factors++; /* the radix  */
  const int m = *factors++; /* stage's fft length/p */

  if (m == 1) {
    int j;
    for (j = 0; j < p; j++) {
      Fout[j] = *f;
      f += fstride * in_stride;
    }
  } else {
    int j;
    for (j = 0; j < p; j++) {
      kf_shuffle(Fout, f, fstride * p, in_stride, factors, st);
      f += fstride * in_stride;
      Fout += m;
    }
  }
}

static void kf_work(kiss_fft_cpx* Fout,
                    const kiss_fft_cpx* f,
                    const size_t fstride,
                    int in_stride,
                    int* factors,
                    const kiss_fft_cfg st,
                    int N,
                    int s2,
                    int m2) {
  int i;
  kiss_fft_cpx* Fout_beg = Fout;
  const int p = *factors++; /* the radix  */
  const int m = *factors++; /* stage's fft length/p */
  if (m == 1) {
  } else {
    kf_work(Fout, f, fstride * p, in_stride, factors, st, N * p,
            fstride * in_stride, m);
  }
  switch (p) {
    case 2:
      kf_bfly2(Fout, fstride, st, m, N, m2);
      break;
    case 3:
      for (i = 0; i < N; i++) {
        Fout = Fout_beg + i * m2;
        kf_bfly3(Fout, fstride, st, m);
      }
      break;
    case 4:
      kf_bfly4(Fout, fstride, st, m, N, m2);
      break;
    case 5:
      for (i = 0; i < N; i++) {
        Fout = Fout_beg + i * m2;
        kf_bfly5(Fout, fstride, st, m);
      }
      break;
    default:
      for (i = 0; i < N; i++) {
        Fout = Fout_beg + i * m2;
        kf_bfly_generic(Fout, fstride, st, m, p);
      }
      break;
  }
}

static void kf_factor(int n, int* facbuf) {
  int p = 4;

  /*factor out powers of 4, powers of 2, then any remaining primes */
  do {
    while (n % p) {
      switch (p) {
        case 4:
          p = 2;
          break;
        case 2:
          p = 3;
          break;
        default:
          p += 2;
          break;
      }
      if (p > 32000 || (int)p * (int)p > n)
        p = n; /* no more factors, skip to end */
    }
    n /= p;
    *facbuf++ = p;
    *facbuf++ = n;
  } while (n > 1);
}

kiss_fft_cfg kiss_fft_alloc(int nfft,
                            int inverse_fft,
                            void* mem,
                            size_t* lenmem) {
  kiss_fft_cfg st = NULL;
  size_t memneeded = sizeof(struct kiss_fft_state) +
                     sizeof(kiss_fft_cpx) * (nfft - 1); /* twiddle factors*/

  if (lenmem == NULL) {
    st = (kiss_fft_cfg)KISS_FFT_MALLOC(memneeded);
  } else {
    if (mem != NULL && *lenmem >= memneeded)
      st = (kiss_fft_cfg)mem;
    *lenmem = memneeded;
  }
  if (st) {
    int i;
    st->nfft = nfft;
    st->inverse = inverse_fft;
    for (i = 0; i < nfft; ++i) {
      const double pi = 3.14159265358979323846264338327;
      double phase = (-2 * pi / nfft) * i;
      if (st->inverse)
        phase *= -1;
      kf_cexp(st->twiddles + i, phase);
    }
    kf_factor(nfft, st->factors);
  }
  return st;
}

void kiss_fft_stride(kiss_fft_cfg st,
                     const kiss_fft_cpx* fin,
                     kiss_fft_cpx* fout,
                     int in_stride) {
  if (fin == fout) {
  } else {
    kf_shuffle(fout, fin, 1, in_stride, st->factors, st);
    kf_work(fout, fin, 1, in_stride, st->factors, st, 1, in_stride, 1);
  }
}

void kiss_fft(kiss_fft_cfg cfg, const kiss_fft_cpx* fin, kiss_fft_cpx* fout) {
  kiss_fft_stride(cfg, fin, fout, 1);
}

struct kiss_fftr_state {
  kiss_fft_cfg substate;
  kiss_fft_cpx* tmpbuf;
  kiss_fft_cpx* super_twiddles;
#ifdef USE_SIMD
  long pad;
#endif
};

kiss_fftr_cfg kiss_fftr_alloc(int nfft,
                              int inverse_fft,
                              void* mem,
                              size_t* lenmem) {
  int i;
  kiss_fftr_cfg st = NULL;
  size_t subsize, memneeded;

  nfft >>= 1;

  kiss_fft_alloc(nfft, inverse_fft, NULL, &subsize);
  memneeded = sizeof(struct kiss_fftr_state) + subsize +
              sizeof(kiss_fft_cpx) * (nfft * 2);

  if (lenmem == NULL) {
    st = (kiss_fftr_cfg)KISS_FFT_MALLOC(memneeded);
  } else {
    if (*lenmem >= memneeded)
      st = (kiss_fftr_cfg)mem;
    *lenmem = memneeded;
  }
  if (!st)
    return NULL;

  st->substate = (kiss_fft_cfg)(st + 1); /*just beyond kiss_fftr_state struct */
  st->tmpbuf = (kiss_fft_cpx*)(((char*)st->substate) + subsize);
  st->super_twiddles = st->tmpbuf + nfft;
  kiss_fft_alloc(nfft, inverse_fft, st->substate, &subsize);

  for (i = 0; i < nfft; ++i) {
    const double pi = 3.14159265358979323846264338327;
    double phase = pi * (((double)i) / nfft + .5);
    if (!inverse_fft)
      phase = -phase;
    kf_cexp(st->super_twiddles + i, phase);
  }
  return st;
}

void kiss_fftr(kiss_fftr_cfg st,
               const kiss_fft_scalar* timedata,
               kiss_fft_cpx* freqdata) {
  /* input buffer timedata is stored row-wise */
  int k, ncfft;
  kiss_fft_cpx fpnk, fpk, f1k, f2k, tw, tdc;

  if (st->substate->inverse) {
    //		speex_fatal("kiss fft usage error: improper alloc\n");
  }

  ncfft = st->substate->nfft;

  /*perform the parallel fft of two real signals packed in real,imag*/
  kiss_fft(st->substate, (const kiss_fft_cpx*)timedata, st->tmpbuf);
  tdc.r = st->tmpbuf[0].r;
  tdc.i = st->tmpbuf[0].i;
  C_FIXDIV(tdc, 2);
  CHECK_OVERFLOW_OP(tdc.r, +, tdc.i);
  CHECK_OVERFLOW_OP(tdc.r, -, tdc.i);
  freqdata[0].r = tdc.r + tdc.i;
  freqdata[ncfft].r = tdc.r - tdc.i;
#ifdef USE_SIMD
  freqdata[ncfft].i = freqdata[0].i = _mm_set1_ps(0);
#else
  freqdata[ncfft].i = freqdata[0].i = 0;
#endif

  for (k = 1; k <= ncfft / 2; ++k) {
    fpk = st->tmpbuf[k];
    fpnk.r = st->tmpbuf[ncfft - k].r;
    fpnk.i = -st->tmpbuf[ncfft - k].i;
    C_FIXDIV(fpk, 2);
    C_FIXDIV(fpnk, 2);

    C_ADD(f1k, fpk, fpnk);
    C_SUB(f2k, fpk, fpnk);
    C_MUL(tw, f2k, st->super_twiddles[k]);

    freqdata[k].r = HALF_OF(f1k.r + tw.r);
    freqdata[k].i = HALF_OF(f1k.i + tw.i);
    freqdata[ncfft - k].r = HALF_OF(f1k.r - tw.r);
    freqdata[ncfft - k].i = HALF_OF(tw.i - f1k.i);
  }
}

void kiss_fftri(kiss_fftr_cfg st,
                const kiss_fft_cpx* freqdata,
                kiss_fft_scalar* timedata) {
  /* input buffer timedata is stored row-wise */
  int k, ncfft;

  ncfft = st->substate->nfft;

  st->tmpbuf[0].r = freqdata[0].r + freqdata[ncfft].r;
  st->tmpbuf[0].i = freqdata[0].r - freqdata[ncfft].r;
  /*C_FIXDIV(st->tmpbuf[0],2);*/

  for (k = 1; k <= ncfft / 2; ++k) {
    kiss_fft_cpx fk, fnkc, fek, fok, tmp;
    fk = freqdata[k];
    fnkc.r = freqdata[ncfft - k].r;
    fnkc.i = -freqdata[ncfft - k].i;
    /*C_FIXDIV( fk , 2 );
    C_FIXDIV( fnkc , 2 );*/

    C_ADD(fek, fk, fnkc);
    C_SUB(tmp, fk, fnkc);
    C_MUL(fok, tmp, st->super_twiddles[k]);
    C_ADD(st->tmpbuf[k], fek, fok);
    C_SUB(st->tmpbuf[ncfft - k], fek, fok);
#ifdef USE_SIMD
    st->tmpbuf[ncfft - k].i *= _mm_set1_ps(-1.0);
#else
    st->tmpbuf[ncfft - k].i *= -1;
#endif
  }
  kiss_fft(st->substate, st->tmpbuf, (kiss_fft_cpx*)timedata);
}

void kiss_fftr2(kiss_fftr_cfg st,
                const kiss_fft_scalar* timedata,
                kiss_fft_scalar* freqdata) {
  /* input buffer timedata is stored row-wise */
  int k, ncfft;
  kiss_fft_cpx f2k, tdc;
  float f1kr, f1ki, twr, twi;

  ncfft = st->substate->nfft;

  /*perform the parallel fft of two real signals packed in real,imag*/
  kiss_fft(st->substate, (const kiss_fft_cpx*)timedata, st->tmpbuf);

  tdc.r = st->tmpbuf[0].r;
  tdc.i = st->tmpbuf[0].i;
  C_FIXDIV(tdc, 2);
  CHECK_OVERFLOW_OP(tdc.r, +, tdc.i);
  CHECK_OVERFLOW_OP(tdc.r, -, tdc.i);
  freqdata[0] = tdc.r + tdc.i;
  freqdata[2 * ncfft - 1] = tdc.r - tdc.i;

  for (k = 1; k <= ncfft / 2; ++k) {
    f2k.r = SHR32(
        SUB32(EXTEND32(st->tmpbuf[k].r), EXTEND32(st->tmpbuf[ncfft - k].r)), 1);
    f2k.i = PSHR32(
        ADD32(EXTEND32(st->tmpbuf[k].i), EXTEND32(st->tmpbuf[ncfft - k].i)), 1);

    f1kr = SHL32(
        ADD32(EXTEND32(st->tmpbuf[k].r), EXTEND32(st->tmpbuf[ncfft - k].r)),
        13);
    f1ki = SHL32(
        SUB32(EXTEND32(st->tmpbuf[k].i), EXTEND32(st->tmpbuf[ncfft - k].i)),
        13);

    twr = SHR32(SUB32(MULT16_16(f2k.r, st->super_twiddles[k].r),
                      MULT16_16(f2k.i, st->super_twiddles[k].i)),
                1);
    twi = SHR32(ADD32(MULT16_16(f2k.i, st->super_twiddles[k].r),
                      MULT16_16(f2k.r, st->super_twiddles[k].i)),
                1);

    freqdata[2 * k - 1] = .5f * (f1kr + twr);
    freqdata[2 * k] = .5f * (f1ki + twi);
    freqdata[2 * (ncfft - k) - 1] = .5f * (f1kr - twr);
    freqdata[2 * (ncfft - k)] = .5f * (twi - f1ki);
  }
}

void kiss_fftri2(kiss_fftr_cfg st,
                 const kiss_fft_scalar* freqdata,
                 kiss_fft_scalar* timedata) {
  /* input buffer timedata is stored row-wise */
  int k, ncfft;

  ncfft = st->substate->nfft;

  st->tmpbuf[0].r = freqdata[0] + freqdata[2 * ncfft - 1];
  st->tmpbuf[0].i = freqdata[0] - freqdata[2 * ncfft - 1];
  /*C_FIXDIV(st->tmpbuf[0],2);*/

  for (k = 1; k <= ncfft / 2; ++k) {
    kiss_fft_cpx fk, fnkc, fek, fok, tmp;
    fk.r = freqdata[2 * k - 1];
    fk.i = freqdata[2 * k];
    fnkc.r = freqdata[2 * (ncfft - k) - 1];
    fnkc.i = -freqdata[2 * (ncfft - k)];
    /*C_FIXDIV( fk , 2 );
    C_FIXDIV( fnkc , 2 );*/

    C_ADD(fek, fk, fnkc);
    C_SUB(tmp, fk, fnkc);
    C_MUL(fok, tmp, st->super_twiddles[k]);
    C_ADD(st->tmpbuf[k], fek, fok);
    C_SUB(st->tmpbuf[ncfft - k], fek, fok);
#ifdef USE_SIMD
    st->tmpbuf[ncfft - k].i *= _mm_set1_ps(-1.0);
#else
    st->tmpbuf[ncfft - k].i *= -1;
#endif
  }
  kiss_fft(st->substate, st->tmpbuf, (kiss_fft_cpx*)timedata);
}

#define MAX_FFT_SIZE 2048

void* linear_fft_init(int size) {
  struct kiss_config* table;
  table = (struct kiss_config*)linear_alloc(sizeof(struct kiss_config));
  table->forward = kiss_fftr_alloc(size, 0, NULL, NULL);
  table->backward = kiss_fftr_alloc(size, 1, NULL, NULL);
  table->N = size;
  return table;
}

void linear_fft_destroy(void* table) {
  struct kiss_config* t = (struct kiss_config*)table;
  kiss_fftr_free(t->forward);
  kiss_fftr_free(t->backward);
  linear_free(table);
}

void linear_fft(void* table, float* in, float* out) {
  int i;
  float scale;
  struct kiss_config* t = (struct kiss_config*)table;
  scale = 1. / t->N;
  kiss_fftr2(t->forward, in, out);
  for (i = 0; i < t->N; i++)
    out[i] *= scale;
}

void linear_ifft(void* table, float* in, float* out) {
  struct kiss_config* t = (struct kiss_config*)table;
  kiss_fftri2(t->backward, in, out);
}

/* This inner product is slightly different from the codec version because of
 * fixed-point */
static inline float mdf_inner_prod(const float* x, const float* y, int len) {
  float sum = 0;
  len >>= 1;
  while (len--) {
    float part = 0;
    part = MAC16_16(part, *x++, *y++);
    part = MAC16_16(part, *x++, *y++);
    /* HINT: If you had a 40-bit accumulator, you could shift only at the end */
    sum = sum + part;
  }
  return sum;
}

/** Compute power spectrum of a half-complex (packed) vector */
static inline void power_spectrum(const float* X, float* ps, int N) {
  int i, j;
  ps[0] = X[0] * X[0];
  for (i = 1, j = 1; i < N - 1; i += 2, j++) {
    ps[j] = X[i] * X[i] + X[i + 1] * X[i + 1];
  }
  ps[j] = X[i] * X[i];
}

/** Compute cross-power spectrum of a half-complex (packed) vectors and add to
 * acc */
static inline void spectral_mul_accum(const float* X,
                                      const float* Y,
                                      float* acc,
                                      int N,
                                      int M) {
  int i, j;
  for (i = 0; i < N; i++) {
    acc[i] = 0;
  }
  for (j = 0; j < M; j++) {
    acc[0] += X[0] * Y[0];
    for (i = 1; i < N - 1; i += 2) {
      acc[i] += (X[i] * Y[i] - X[i + 1] * Y[i + 1]);
      acc[i + 1] += (X[i + 1] * Y[i] + X[i] * Y[i + 1]);
    }
    acc[i] += X[i] * Y[i];
    X += N;
    Y += N;
  }
}

/** Compute weighted cross-power spectrum of a half-complex (packed) vector with
 * conjugate */
static inline void weighted_spectral_mul_conj(const float* w,
                                              const float p,
                                              const float* X,
                                              const float* Y,
                                              float* prod,
                                              int N) {
  int i, j;
  float W;
  W = (p * w[0]);
  prod[0] = (W * (X[0] * Y[0]));
  for (i = 1, j = 1; i < N - 1; i += 2, j++) {
    W = (p * w[j]);
    prod[i] = (W * ((X[i] * Y[i]) + X[i + 1] * Y[i + 1]));
    prod[i + 1] = (W * ((-X[i + 1] * Y[i]) + X[i] * Y[i + 1]));
  }
  W = (p * w[j]);
  prod[i] = (W * (X[i] * Y[i]));
}

static inline void mdf_adjust_prop(const float* W, int N, int M, float* prop) {
  int i, j;
  float max_sum = 1;
  float prop_sum = 1;
  for (i = 0; i < M; i++) {
    float tmp = 1;
    for (j = 0; j < N; j++) {
      tmp += W[i * N + j] * W[i * N + j];
    }

    prop[i] = (float)(sqrt(tmp));
    if (prop[i] > max_sum) {
      max_sum = prop[i];
    }
  }
  for (i = 0; i < M; i++) {
    prop[i] += 0.1f * max_sum;
    prop_sum += prop[i];
  }
  for (i = 0; i < M; i++) {
    prop[i] = 0.99 * prop[i] / prop_sum;
  }
}

/** Creates a new echo canceller state */
LinearEchoState* linear_echo_state_init(int frame_size,
                                        int filter_length,
                                        int samplerate) {
  int i, N, M;
  LinearEchoState* st = (LinearEchoState*)linear_alloc(sizeof(LinearEchoState));

  st->frame_size = frame_size;
  st->window_size = 2 * frame_size;
  N = st->window_size;
  M = st->M = (filter_length + st->frame_size - 1) / frame_size;
  st->cancel_count = 0;
  st->sum_adapt = 0;
  st->saturated = 0;
  st->screwed_up = 0;
  /* This is the default sampling rate */
  st->sampling_rate = samplerate;
  st->spec_average = st->frame_size * 1.0f / st->sampling_rate;
  st->beta0 = (2.0f * st->frame_size) / st->sampling_rate;
  st->beta_max = (.5f * st->frame_size) / st->sampling_rate;
  st->leak_estimate = 0;
  st->fft_table = linear_fft_init(N);

  // st->e = (float*)linear_alloc(N * sizeof(float));
  // st->x = (float*)linear_alloc(N * sizeof(float));
  // st->input = (float*)linear_alloc(st->frame_size * sizeof(float));
  // st->y = (float*)linear_alloc(N * sizeof(float));
  // st->last_y = (float*)linear_alloc(N * sizeof(float));
  // st->Yf = (float*)linear_alloc((st->frame_size + 1) * sizeof(float));
  // st->Rf = (float*)linear_alloc((st->frame_size + 1) * sizeof(float));
  // st->Xf = (float*)linear_alloc((st->frame_size + 1) * sizeof(float));
  // st->Yh = (float*)linear_alloc((st->frame_size + 1) * sizeof(float));
  // st->Eh = (float*)linear_alloc((st->frame_size + 1) * sizeof(float));

  // st->X = (float*)linear_alloc((M + 1)*N * sizeof(float));
  // st->Y = (float*)linear_alloc(N * sizeof(float));
  // st->E = (float*)linear_alloc(N * sizeof(float));
  // st->W = (float*)linear_alloc(M*N * sizeof(float));
  // st->foreground = (float*)linear_alloc(M*N * sizeof(float));

  // st->PHI = (float*)linear_alloc(N * sizeof(float));
  // st->power = (float*)linear_alloc((frame_size + 1) * sizeof(float));
  // st->power_1 = (float*)linear_alloc((frame_size + 1) * sizeof(float));
  // st->window = (float*)linear_alloc(N * sizeof(float));
  // st->prop = (float*)linear_alloc(M * sizeof(float));
  // st->wtmp = (float*)linear_alloc(N * sizeof(float));

  /*for (i = 0; i < N; i++)
          st->window[i] = .5 - .5*cos(2 * M_PI * i / N);*/

  for (i = 0; i <= st->frame_size; i++)
    st->power_1[i] = 1.0f;
  for (i = 0; i < N * M; i++)
    st->W[i] = 0;
  {
    float sum = 0;
    /* Ratio of ~10 between adaptation rate of first and last block */
    float decay = exp(NEG16(2.4 / M));
    st->prop[0] = 0.70f;
    sum = st->prop[0];
    for (i = 1; i < M; i++) {
      st->prop[i] = (st->prop[i - 1] * decay);
      sum = sum + st->prop[i];
    }
    for (i = M - 1; i >= 0; i--) {
      st->prop[i] = 0.80f * st->prop[i] / sum;
    }
  }

  st->memX = st->memD = st->memE = 0;
  st->memLinear = 0.0;
  st->preemph = 0.90f;
  if (st->sampling_rate < 12000) {
    st->notch_radius = 0.90f;
  } else if (st->sampling_rate < 24000) {
    st->notch_radius = 0.982f;
  } else {
    st->notch_radius = 0.992f;
  }

  st->notch_mem[0] = st->notch_mem[1] = 0;
  st->updateflag = 0;
  st->resetflag = 0;
  st->adapted = 0;
  st->Pey = st->Pyy = 1.0f;

  st->Davg1 = st->Davg2 = 0;
  st->Dvar1 = st->Dvar2 = 0.0f;

#ifdef LINEAR_FILTER_ROCORD
  st->f_linear_near_in = NULL;
  st->f_linear_far_in = NULL;
  st->f_linear_near_out = NULL;
  st->f_fft_in = NULL;
  st->f_fft_out = NULL;
  st->f_ifft_in = NULL;
  st->f_ifft_out = NULL;
  st->f_near_in = NULL;
  st->f_far_in = NULL;
  if (st->f_linear_near_in == NULL) {
    st->f_linear_near_in = fopen("linear_near_in.pcm", "wb");
  }
  if (st->f_linear_far_in == NULL) {
    st->f_linear_far_in = fopen("linear_far_in.pcm", "wb");
  }
  if (st->f_linear_near_out == NULL) {
    st->f_linear_near_out = fopen("linear_near_out.pcm", "wb");
  }
  if (st->f_fft_in == NULL) {
    st->f_fft_in = fopen("fft_in.pcm", "wb");
  }
  if (st->f_fft_out == NULL) {
    st->f_fft_out = fopen("fft_out.pcm", "wb");
  }
  if (st->f_ifft_in == NULL) {
    st->f_ifft_in = fopen("ifft_in.pcm", "wb");
  }
  if (st->f_ifft_out == NULL) {
    st->f_ifft_out = fopen("ifft_out.pcm", "wb");
  }
  if (st->f_near_in == NULL) {
    st->f_near_in = fopen("near.pcm", "wb");
  }
  if (st->f_far_in == NULL) {
    st->f_far_in = fopen("far.pcm", "wb");
  }
#endif

  return st;
}

/** Resets echo canceller state */
void linear_echo_state_reset(LinearEchoState* st) {
  int i, M, N;
  st->cancel_count = 0;
  st->screwed_up = 0;
  N = st->window_size;
  M = st->M;
  for (i = 0; i < N * M; i++)
    st->W[i] = 0;

  for (i = 0; i < N * M; i++)
    st->foreground[i] = 0;

  for (i = 0; i < N * (M + 1); i++)
    st->X[i] = 0;
  for (i = 0; i <= st->frame_size; i++) {
    st->power[i] = 0;
    st->power_1[i] = 1.0f;
    st->Eh[i] = 0;
    st->Yh[i] = 0;
  }
  for (i = 0; i < st->frame_size; i++) {
    st->last_y[i] = 0;
  }
  for (i = 0; i < N; i++) {
    st->E[i] = 0;
    st->x[i] = 0;
  }
  st->notch_mem[0] = st->notch_mem[1] = 0;
  st->updateflag = 0;
  st->resetflag = 0;
  st->memX = st->memD = st->memE = 0;
  st->memLinear = 0.0;
  st->saturated = 0;
  st->adapted = 0;
  st->sum_adapt = 0;
  st->Pey = st->Pyy = 1.0f;

  st->Davg1 = st->Davg2 = 0;
  st->Dvar1 = st->Dvar2 = 0.0f;
}

/** Destroys an echo canceller state */
void linear_echo_state_destroy(LinearEchoState* st) {
  linear_fft_destroy(st->fft_table);

  // linear_free(st->e);
  // linear_free(st->x);
  // linear_free(st->input);
  // linear_free(st->y);
  // linear_free(st->last_y);
  // linear_free(st->Yf);
  // linear_free(st->Rf);
  // linear_free(st->Xf);
  // linear_free(st->Yh);
  // linear_free(st->Eh);

  // linear_free(st->X);
  // linear_free(st->Y);
  // linear_free(st->E);
  // linear_free(st->W);
  // linear_free(st->foreground);

  // linear_free(st->PHI);
  // linear_free(st->power);
  // linear_free(st->power_1);
  // linear_free(st->window);
  // linear_free(st->prop);
  // linear_free(st->wtmp);
#ifdef LINEAR_FILTER_ROCORD
  if (st->f_linear_near_in != NULL) {
    fclose(st->f_linear_near_in);
    st->f_linear_near_in = NULL;
  }
  if (st->f_linear_far_in != NULL) {
    fclose(st->f_linear_far_in);
    st->f_linear_far_in = NULL;
  }
  if (st->f_linear_near_out != NULL) {
    fclose(st->f_linear_near_out);
    st->f_linear_near_out = NULL;
  }
  if (st->f_fft_in != NULL) {
    fclose(st->f_fft_in);
    st->f_fft_in = NULL;
  }
  if (st->f_fft_out != NULL) {
    fclose(st->f_fft_out);
    st->f_fft_out = NULL;
  }
  if (st->f_ifft_in != NULL) {
    fclose(st->f_ifft_in);
    st->f_ifft_in = NULL;
  }
  if (st->f_ifft_out != NULL) {
    fclose(st->f_ifft_out);
    st->f_ifft_out = NULL;
  }
  if (st->f_near_in != NULL) {
    fclose(st->f_near_in);
    st->f_near_in = NULL;
  }
  if (st->f_far_in != NULL) {
    fclose(st->f_far_in);
    st->f_far_in = NULL;
  }
#endif
  linear_free(st);
}

static inline void filter_dc_notch16(float* in,
                                     float radius,
                                     float* out,
                                     int len,
                                     float* mem) {
  int i;
  float den2;
  den2 = radius * radius + 0.7f * (1 - radius) * (1 - radius);
  for (i = 0; i < len; i++) {
    float vin = in[i];
    float vout = mem[0] + vin;
    mem[0] = mem[1] + 2 * (-vin + radius * vout);
    mem[1] = vin - den2 * vout;
    out[i] = radius * vout;
  }
}

/** Performs echo cancellation on a frame */
void linear_echo_cancellation(LinearEchoState* st,
                              float* in,
                              float* far_end,
                              short* out,
                              short* linear_out,
                              float* leakestimate,
                              short* nlpFlag) {
  int i, j;
  int N, M;
  float Syy, See, Sxx, Sdd, Sff;
  float Dbf;
  int update_foreground;
  float Sey;
  float ss, ss_1;
  float Pey = 1.0f, Pyy = 1.0f;
  float alpha, alpha_1;
  float RER;
  float tmp32;
  // int length = sizeof(st->fft_table);

  N = st->window_size;
  M = st->M;
  st->cancel_count++;
  ss = .35 / M;
  ss_1 = 1 - ss;
#ifdef LINEAR_FILTER_ROCORD
  short temp_far[PART_LEN] = {0};

  for (i = 0; i < PART_LEN; i++) {
    if (far_end[i] > 32767) {
      temp_far[i] = 32767;
    } else if (far_end[i] < -32768) {
      temp_far[i] = -32768;
    } else {
      temp_far[i] = (short)(far_end[i]);
    }
  }
  if (st->f_far_in != NULL) {
    fwrite(temp_far, sizeof(short), PART_LEN, st->f_far_in);
  }
#endif
  /* Apply a notch filter to make sure DC doesn't end up causing problems */
  filter_dc_notch16(in, st->notch_radius, st->input, st->frame_size,
                    st->notch_mem);

  /* Copy input data to buffer and apply pre-emphasis */
  for (i = 0; i < st->frame_size; i++) {
    float tmp32;
    tmp32 = far_end[i] - st->preemph * st->memX;
    st->x[i + st->frame_size] = tmp32;
    st->memX = far_end[i];

    tmp32 = st->input[i] - st->preemph * st->memD;
    st->memD = st->input[i];
    st->input[i] = tmp32;
  }

  /* Shift memory: this could be optimized eventually*/
  for (j = M - 1; j >= 0; j--) {
    for (i = 0; i < N; i++) {
      st->X[(j + 1) * N + i] = st->X[j * N + i];
    }
  }
#ifdef LINEAR_FILTER_ROCORD
  short temp_near[PART_LEN] = {0};

  for (i = 0; i < PART_LEN; i++) {
    if (st->x[i + PART_LEN] > 32767) {
      temp_near[i] = 32767;
    } else if (st->x[i + PART_LEN] < -32768) {
      temp_near[i] = -32768;
    } else {
      temp_near[i] = (short)(st->x[i + PART_LEN]);
    }
  }
  if (st->f_near_in != NULL) {
    fwrite(temp_near, sizeof(short), PART_LEN, st->f_near_in);
  }
#endif
  /* Convert x (far end) to frequency domain */
  linear_fft(st->fft_table, st->x, &st->X[0]);

#ifdef LINEAR_FILTER_ROCORD
  if (st->f_fft_in != NULL) {
    fwrite(st->x, sizeof(float), 128, st->f_fft_in);
  }
  if (st->f_fft_out != NULL) {
    fwrite(&st->X[0], sizeof(float), 128, st->f_fft_out);
  }
#endif
  for (i = 0; i < N; i++) {
    st->last_y[i] = st->x[i];
  }
  Sxx = mdf_inner_prod(st->x + st->frame_size, st->x + st->frame_size,
                       st->frame_size);
  for (i = 0; i < st->frame_size; i++) {
    st->x[i] = st->x[i + st->frame_size];
  }
  /* From here on, the top part of x is used as scratch space */

  /* Compute foreground filter */
  spectral_mul_accum(st->X, st->foreground, st->Y, N, M);
  linear_ifft(st->fft_table, st->Y, st->e);

#ifdef LINEAR_FILTER_ROCORD
  if (st->f_ifft_in != NULL) {
    fwrite(st->Y, sizeof(float), 128, st->f_ifft_in);
  }
  if (st->f_ifft_out != NULL) {
    fwrite(st->e, sizeof(float), 128, st->f_ifft_out);
  }
#endif

  for (i = 0; i < st->frame_size; i++) {
    st->e[i] = st->input[i] - st->e[i + st->frame_size];
  }
  Sff = mdf_inner_prod(st->e, st->e, st->frame_size);

  /* Adjust proportional adaption rate */
  mdf_adjust_prop(st->W, N, M, st->prop);
  /* Compute weight gradient */
  if (st->saturated == 0) {
    for (j = M - 1; j >= 0; j--) {
      weighted_spectral_mul_conj(st->power_1, st->prop[j], &st->X[(j + 1) * N],
                                 st->E, st->PHI, N);
      for (i = 0; i < N; i++) {
        st->W[j * N + i] = st->W[j * N + i] + st->PHI[i];
      }
    }
  } else {
    st->saturated--;
  }

  ///* Update weight to prevent circular convolution (MDF / AUMDF) */
  for (j = 0; j < M; j++) {
    /* This is a variant of the Alternatively Updated MDF (AUMDF) */
    /* Remove the "if" to make this an MDF filter */
    if (j == 0 || st->cancel_count % (M - 1) == j - 1) {
      linear_ifft(st->fft_table, &st->W[j * N], st->wtmp);

      for (i = st->frame_size; i < N; i++) {
        st->wtmp[i] = 0;
      }
      linear_fft(st->fft_table, st->wtmp, &st->W[j * N]);
    }
  }

  /* Compute filter response Y */
  spectral_mul_accum(st->X, st->W, st->Y, N, M);
  linear_ifft(st->fft_table, st->Y, st->y);

  /* Difference in response, this is used to estimate the variance of our
   * residual power estimate */
  for (i = 0; i < st->frame_size; i++) {
    st->e[i] = st->e[i + st->frame_size] - st->y[i + st->frame_size];
  }
  Dbf = 10 + mdf_inner_prod(st->e, st->e, st->frame_size);

  for (i = 0; i < st->frame_size; i++) {
    st->e[i] = st->input[i] - st->y[i + st->frame_size];
  }
  See = mdf_inner_prod(st->e, st->e, st->frame_size);

  /* Logic for updating the foreground filter */

  /* For two time windows, compute the mean of the energy difference, as well as
   * the variance */
  st->Davg1 = 0.6 * st->Davg1 + 0.4 * (Sff - See);
  st->Davg2 = 0.85 * st->Davg2 + 0.15 * (Sff - See);
  st->Dvar1 = 0.36 * st->Dvar1 + 0.16 * Sff * Dbf;
  st->Dvar2 = 0.7225 * st->Dvar2 + 0.0225 * Sff * Dbf;

  update_foreground = 0;
  /* Check if we have a statistically significant reduction in the residual echo
   */
  /* Note that this is *not* Gaussian, so we need to be careful about the longer
   * tail */
  if (FLOAT_GT(((Sff - See) * ABS32((Sff - See))), (Sff * Dbf))) {
    update_foreground = 1;
  } else if (FLOAT_GT((st->Davg1 * ABS32(st->Davg1)),
                      (VAR1_UPDATE * (st->Dvar1)))) {
    update_foreground = 1;
  } else if (FLOAT_GT((st->Davg2 * ABS32(st->Davg2)),
                      (VAR2_UPDATE * (st->Dvar2)))) {
    update_foreground = 1;
  }
  st->updateflag = update_foreground;
  /* Do we update? */
  if (update_foreground) {
    st->Davg1 = st->Davg2 = 0;
    st->Dvar1 = st->Dvar2 = 0.0f;
    /* Copy background filter to foreground filter */
    for (i = 0; i < N * M; i++) {
      st->foreground[i] = st->W[i];
    }
    /* Apply a smooth transition so as to not introduce blocking artifacts */
    for (i = 0; i < st->frame_size; i++) {
      st->e[i + st->frame_size] =
          (WebRtcAec_mdfWindows[i + st->frame_size] *
           st->e[i + st->frame_size]) +
          (WebRtcAec_mdfWindows[i] * st->y[i + st->frame_size]);
    }
  } else {
    int reset_background = 0;
    /* Otherwise, check if the background filter is significantly worse */
    if (FLOAT_GT((NEG32((Sff - See)) * ABS32((Sff - See))),
                 (VAR_BACKTRACK * (Sff * Dbf)))) {
      reset_background = 1;
    }
    if (FLOAT_GT((NEG32(st->Davg1) * ABS32(st->Davg1)),
                 (VAR_BACKTRACK * st->Dvar1))) {
      reset_background = 1;
    }
    if (FLOAT_GT((NEG32(st->Davg2) * ABS32(st->Davg2)),
                 (VAR_BACKTRACK * st->Dvar2))) {
      reset_background = 1;
    }
    st->resetflag = reset_background;
    if (reset_background) {
      /* Copy foreground filter to background filter */
      for (i = 0; i < N * M; i++) {
        st->W[i] = st->foreground[i];
      }
      /* We also need to copy the output so as to get correct adaptation */
      for (i = 0; i < st->frame_size; i++) {
        st->y[i + st->frame_size] = st->e[i + st->frame_size];
      }
      for (i = 0; i < st->frame_size; i++) {
        st->e[i] = st->input[i] - st->y[i + st->frame_size];
      }
      See = Sff;
      st->Davg1 = st->Davg2 = 0;
      st->Dvar1 = st->Dvar2 = 0.0f;
    }
  }

  /* Compute error signal (for the output with de-emphasis) */
  for (i = 0; i < st->frame_size; i++) {
    float tmp_out;
    tmp_out = st->input[i] - st->e[i + st->frame_size];
    /* Saturation */
    tmp_out = tmp_out + st->preemph * st->memE;
    /* This is an arbitrary test for saturation in the microphone signal */
    if (in[i] <= -32000 || in[i] >= 32000) {
      if (st->saturated == 0) {
        st->saturated = 0;
      }
    }
    if (tmp_out > 32767) {
      out[i] = 32767;
    } else if (tmp_out < -32768) {
      out[i] = -32768;
    } else {
      out[i] = (short)(tmp_out * 1);
    }
    st->memE = tmp_out;
  }

  /* Compute Filter output signal (for the output with de-emphasis) */
  for (i = 0; i < st->frame_size; i++) {
    float tmp_out;
    tmp_out = st->e[i + st->frame_size];
    /* Saturation */
    tmp_out = tmp_out + st->preemph * st->memLinear;

    if (tmp_out > 32767) {
      linear_out[i] = 32767;
    } else if (tmp_out < -32768) {
      linear_out[i] = -32768;
    } else {
      linear_out[i] = (short)(tmp_out * 1);
    }
    st->memLinear = tmp_out;
  }

  /* Compute error signal (filter update version) */
  for (i = 0; i < st->frame_size; i++) {
    st->e[i + st->frame_size] = st->e[i];
    st->e[i] = 0;
  }

  /* Compute a bunch of correlations */
  Sey = mdf_inner_prod(st->e + st->frame_size, st->y + st->frame_size,
                       st->frame_size);
  Syy = mdf_inner_prod(st->y + st->frame_size, st->y + st->frame_size,
                       st->frame_size);
  Sdd = mdf_inner_prod(st->input, st->input, st->frame_size);

  /* Do some sanity check */
  if (!(Syy >= 0 && Sxx >= 0 && See >= 0) ||
      !(Sff < N * 1e9 && Syy < N * 1e9 && Sxx < N * 1e9)) {
    /* Things have gone really bad */
    st->screwed_up += 50;
  } else if (Sff > (Sdd + N * 10000)) {
    /* AEC seems to add lots of echo instead of removing it, let's see if it
     * will improve */
    st->screwed_up++;
  } else {
    /* Everything's fine */
    if (st->screwed_up > 0) {
      st->screwed_up--;
    } else {
      st->screwed_up = 0;
    }
    // st->screwed_up = 0;
  }

  if (st->screwed_up >= 25) {
    *nlpFlag = 2;
  } else {
    if (st->screwed_up > 0) {
      *nlpFlag = 1;
    } else {
      *nlpFlag = 0;
    }
  }

  if (st->screwed_up >= 50) {
    // linear_echo_state_reset(st);
    // return;
  }

  /* Add a small noise floor to make sure not to have problems when dividing */
  See = MAX32(See, (N * 100));

  /* Convert error to frequency domain */
  linear_fft(st->fft_table, st->e, st->E);

  for (i = 0; i < st->frame_size; i++) {
    st->y[i] = 0;
  }
  linear_fft(st->fft_table, st->y, st->Y);

  /* Compute power spectrum of far end (X), error (E) and filter response (Y) */
  power_spectrum(st->E, st->Rf, N);
  power_spectrum(st->Y, st->Yf, N);
  power_spectrum(st->X, st->Xf, N);

  /* Smooth far end energy estimate over time */
  for (j = 0; j <= st->frame_size; j++) {
    st->power[j] = (ss_1 * st->power[j]) + 1 + (ss * st->Xf[j]);
  }

  /* Compute filtered spectra and (cross-)correlations */
  for (j = st->frame_size; j >= 0; j--) {
    float Eh, Yh;
    Eh = (st->Rf[j] - st->Eh[j]);
    Yh = (st->Yf[j] - st->Yh[j]);
    Pey = (Pey + (Eh * Yh));
    Pyy = (Pyy + (Yh * Yh));
    st->Eh[j] =
        (1 - st->spec_average) * st->Eh[j] + st->spec_average * st->Rf[j];
    st->Yh[j] =
        (1 - st->spec_average) * st->Yh[j] + st->spec_average * st->Yf[j];
  }

  Pyy = sqrt(Pyy);
  Pey = Pey / Pyy;

  /* Compute correlation updatete rate */
  tmp32 = (st->beta0 * Syy);
  if (tmp32 > (st->beta_max * See)) {
    tmp32 = (st->beta_max * See);
  }
  alpha = tmp32 / See;
  alpha_1 = (1.0f - alpha);
  /* Update correlations (recursive average) */
  st->Pey = alpha_1 * st->Pey + alpha * Pey;
  st->Pyy = alpha_1 * st->Pyy + alpha * Pyy;
  if (st->Pyy < 1.0f) {
    st->Pyy = 1.0f;
  }
  /* We don't really hope to get better than 33 dB (MIN_LEAK-3dB) attenuation
   * anyway */
  if (st->Pey < (MIN_LEAK * st->Pyy)) {
    st->Pey = MIN_LEAK * st->Pyy;
  }
  if (st->Pey > st->Pyy) {
    st->Pey = st->Pyy;
  }
  /* leak_estimate is the linear regression result */
  st->leak_estimate = st->Pey / st->Pyy;
  /* This looks like a stupid bug, but it's right (because we convert from Q14
   * to Q15) */
  if (st->leak_estimate > 16383) {
    st->leak_estimate = 32767;
  } else {
    st->leak_estimate = st->leak_estimate;
  }
  *leakestimate = st->leak_estimate;
  /* Compute Residual to Error Ratio */
  RER = (.0001 * Sxx + 3. * (st->leak_estimate * Syy)) / See;
  /* Check for y in e (lower bound on RER) */
  if (RER < Sey * Sey / (1 + See * Syy)) {
    RER = Sey * Sey / (1 + See * Syy);
  }
  if (RER > .5) {
    RER = .5;
  }

  /* We consider that the filter has had minimal adaptation if the following is
   * true*/
  if (!st->adapted && st->sum_adapt > M &&
      (st->leak_estimate * Syy) > (0.03f * Syy)) {
    st->adapted = 1;
  }

  if (st->adapted) {
    /* Normal learning rate calculation once we're past the minimal adaptation
     * phase */
    for (i = 0; i <= st->frame_size; i++) {
      float r, e;
      /* Compute frequency-domain adaptation mask */
      r = (st->leak_estimate * st->Yf[i]);
      e = st->Rf[i] + 1;
      if (r > 0.5 * e) {
        r = 0.5 * e;
      }
      r = 0.7f * r + 0.3f * RER * e;
      st->power_1[i] = r / (e * (st->power[i] + 10));
    }
  } else {
    /* Temporary adaption rate if filter is not yet adapted enough */
    float adapt_rate = 0;

    if (Sxx > (N * 1000)) {
      tmp32 = 0.25f * Sxx;
      if (tmp32 > 0.25 * See) {
        tmp32 = 0.25 * See;
      }
      adapt_rate = tmp32 / See;
    }

    if (adapt_rate < 0.5f) {
      adapt_rate = 0.5f;
    }
    for (i = 0; i <= st->frame_size; i++) {
      st->power_1[i] = adapt_rate / (st->power[i] + 10);
    }
    /* How much have we adapted so far? */
    st->sum_adapt = st->sum_adapt + adapt_rate;
  }

  /* Save residual echo so it can be used by the nonlinear processor */
  if (st->adapted) {
    /* If the filter is adapted, take the filtered echo */
    for (i = 0; i < st->frame_size; i++) {
      st->last_y[i] = st->last_y[st->frame_size + i];
    }
    for (i = 0; i < st->frame_size; i++) {
      st->last_y[st->frame_size + i] = in[i] - out[i];
    }
  } else {
    /* If filter isn't adapted yet, all we can do is take the far end signal
     * directly */
    /* moved earlier: for (i=0;i<N;i++)
    st->last_y[i] = st->x[i];*/
  }
}

void AEC_Process_Core(AecCore* aecpc,
                      float farend_extended_block_lowest_band[PART_LEN2],
                      float nearend_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN],
                      float output_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN]) {
  short error[PART_LEN] = {0};
  float linear_near_in[PART_LEN] = {0.0};
  float linear_far_in[PART_LEN] = {0.0};
  float echo_subtractor_output[PART_LEN] = {0.0};
  short filter_output[PART_LEN] = {0};
  // float linear_filter_output[PART_LEN] = { 0.0 };
  short near_out[128] = {0};
  float nearend_extended_block_lowest_band[PART_LEN2];
  short i = 0;
  size_t index = 0;
  float leakestimate = 0.0;
  short nlpFlag = 0;
  for (i = 0; i < PART_LEN; i++) {
    linear_near_in[i] = nearend_block[0][i];
    linear_far_in[i] = farend_extended_block_lowest_band[i + PART_LEN];
  }
#ifdef LINEAR_FILTER_ROCORD
  short temp_near[PART_LEN] = {0};
  short temp_far[PART_LEN] = {0};

  for (i = 0; i < PART_LEN; i++) {
    if (linear_near_in[i] > 32767) {
      temp_near[i] = 32767;
    } else if (linear_near_in[i] < -32768) {
      temp_near[i] = -32768;
    } else {
      temp_near[i] = (short)(linear_near_in[i]);
    }

    if (linear_far_in[i] > 32767) {
      temp_far[i] = 32767;
    } else if (linear_far_in[i] < -32768) {
      temp_far[i] = -32768;
    } else {
      temp_far[i] = (short)(linear_far_in[i]);
    }
  }
  if (aecpc->st->f_linear_near_in != NULL) {
    fwrite(temp_near, sizeof(short), PART_LEN, aecpc->st->f_linear_near_in);
  }
  if (aecpc->st->f_linear_far_in != NULL) {
    fwrite(temp_far, sizeof(short), PART_LEN, aecpc->st->f_linear_far_in);
  }
#endif
  linear_echo_cancellation(aecpc->st, linear_near_in, linear_far_in, error,
                           filter_output, &leakestimate, &nlpFlag);
  for (i = 0; i < PART_LEN; i++) {
    echo_subtractor_output[i] = error[i];
    // linear_filter_output[i] = filter_output[i];
  }
#ifdef LINEAR_FILTER_ROCORD
  if (aecpc->st->f_linear_near_out != NULL) {
    fwrite(error, sizeof(short), PART_LEN, aecpc->st->f_linear_near_out);
  }
#endif
#ifdef AEC_ROCORD
  if (aecpc->f_aec_linear_out != NULL) {
    fwrite(filter_output, 2, PART_LEN, aecpc->f_aec_linear_out);
  }
#endif
  if (aecpc->metricsMode == 1) {
    UpdateLevel(&aecpc->linoutlevel,
                CalculatePower(echo_subtractor_output, PART_LEN));
  }

  // Form extended nearend frame.
  memcpy(&nearend_extended_block_lowest_band[0],
         &aecpc->previous_nearend_block[0][0], sizeof(float) * PART_LEN);
  memcpy(&nearend_extended_block_lowest_band[PART_LEN], &nearend_block[0][0],
         sizeof(float) * PART_LEN);

  // Perform echo suppression.
  aecpc->nlpFlag = nlpFlag;
  EchoSuppression(
      aecpc->ooura_fft, aecpc, leakestimate, nearend_extended_block_lowest_band,
      farend_extended_block_lowest_band, echo_subtractor_output, output_block);

  for (i = 0; i < PART_LEN; i++) {
    near_out[2 * i] = (short)(nearend_block[0][i]);
    near_out[2 * i + 1] = (short)(output_block[0][i]);
    // output_block[0][i] = error[i];
  }

  // Store the nearend signal until the next frame.
  for (index = 0; index < aecpc->num_bands; ++index) {
    memcpy(&aecpc->previous_nearend_block[index][0], &nearend_block[index][0],
           sizeof(float) * PART_LEN);
  }
  return;
}

void ProcessNearendBlockMdf(
    AecCore* aec,
    int num_partitions,
    int* x_fft_buf_block_pos,
    float farend_extended_block_lowest_band[PART_LEN2],
    float nearend_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN],
    float output_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN]) {
  float temp_near_in[NUM_HIGH_BANDS_MAX + 1][PART_LEN];
  // short foregroundFilter_out[PART_LEN] = { 0 };
  // short backgroundFilter_out[PART_LEN] = { 0 };
  // float tempforeout[PART_LEN] = { 0.0 };
  // float tempbackout[PART_LEN] = { 0.0 };
  short i = 0;
  short j = 0;
  for (i = 0; i < PART_LEN; i++) {
    temp_near_in[0][i] = nearend_block[0][i];
    temp_near_in[1][i] = nearend_block[1][i];
    temp_near_in[2][i] = nearend_block[2][i];
  }

  for (i = 0; i < COUNTER - 1; i++) {
    for (j = 0; j < PART_LEN; j++) {
      aec->near_buf[0][i * PART_LEN + j] =
          aec->near_buf[0][(i + 1) * PART_LEN + j];
      aec->near_buf[1][i * PART_LEN + j] =
          aec->near_buf[1][(i + 1) * PART_LEN + j];
      aec->near_buf[2][i * PART_LEN + j] =
          aec->near_buf[2][(i + 1) * PART_LEN + j];
    }
  }
  for (i = 0; i < PART_LEN; i++) {
    aec->near_buf[0][(COUNTER - 1) * PART_LEN + i] = temp_near_in[0][i];
    aec->near_buf[1][(COUNTER - 1) * PART_LEN + i] = temp_near_in[1][i];
    aec->near_buf[2][(COUNTER - 1) * PART_LEN + i] = temp_near_in[2][i];
  }
  for (i = 0; i < PART_LEN; i++) {
    temp_near_in[0][i] = aec->near_buf[0][i];
    temp_near_in[1][i] = aec->near_buf[1][i];
    temp_near_in[2][i] = aec->near_buf[2][i];
  }
#ifdef AEC_ROCORD
  short temp_far_in[PART_LEN] = {0};
  for (i = 0; i < PART_LEN; i++) {
    if (farend_extended_block_lowest_band[i + PART_LEN] > 32767) {
      temp_far_in[i] = 32767;
    } else if (farend_extended_block_lowest_band[i + PART_LEN] < -32768) {
      temp_far_in[i] = -32768;
    } else {
      temp_far_in[i] = (short)(farend_extended_block_lowest_band[i + PART_LEN]);
    }
  }
  if (aec->f_far_in != NULL) {
    fwrite(temp_far_in, 2, PART_LEN, aec->f_far_in);
  }
#endif
  AEC_Process_Core(aec, farend_extended_block_lowest_band, temp_near_in,
                   output_block);
}
#endif

void ProcessNearendBlock(AecCore* aec,
                         float farend_extended_block_lowest_band[PART_LEN2],
                         float nearend_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN],
                         float output_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN]) {
  size_t i;

  float fft[PART_LEN2];
  float nearend_extended_block_lowest_band[PART_LEN2];
  float farend_fft[2][PART_LEN1];
  float nearend_fft[2][PART_LEN1];
  float far_spectrum = 0.0f;
  float near_spectrum = 0.0f;
  float abs_far_spectrum[PART_LEN1];
  float abs_near_spectrum[PART_LEN1];

  const float gPow[2] = {0.9f, 0.1f};

  // Noise estimate constants.
  const int noiseInitBlocks = 500 * aec->mult;
  const float step = 0.1f;
  const float ramp = 1.0002f;
  const float gInitNoise[2] = {0.999f, 0.001f};
#ifndef AUMDF_FILTER
  float echo_subtractor_output[PART_LEN];
#endif

  aec->data_dumper->DumpWav("aec_far", PART_LEN,
                            &farend_extended_block_lowest_band[PART_LEN],
                            std::min(aec->sampFreq, 16000), 1);
  aec->data_dumper->DumpWav("aec_near", PART_LEN, &nearend_block[0][0],
                            std::min(aec->sampFreq, 16000), 1);

  if (aec->metricsMode == 1) {
    // Update power levels
    UpdateLevel(
        &aec->farlevel,
        CalculatePower(&farend_extended_block_lowest_band[PART_LEN], PART_LEN));
    UpdateLevel(&aec->nearlevel,
                CalculatePower(&nearend_block[0][0], PART_LEN));
  }

  // Convert far-end signal to the frequency domain.
  memcpy(fft, farend_extended_block_lowest_band, sizeof(float) * PART_LEN2);
  Fft(aec->ooura_fft, fft, farend_fft);

  // Form extended nearend frame.
  memcpy(&nearend_extended_block_lowest_band[0],
         &aec->previous_nearend_block[0][0], sizeof(float) * PART_LEN);
  memcpy(&nearend_extended_block_lowest_band[PART_LEN], &nearend_block[0][0],
         sizeof(float) * PART_LEN);

  // Convert near-end signal to the frequency domain.
  memcpy(fft, nearend_extended_block_lowest_band, sizeof(float) * PART_LEN2);
  Fft(aec->ooura_fft, fft, nearend_fft);

  // Power smoothing.
  if (aec->refined_adaptive_filter_enabled) {
    for (i = 0; i < PART_LEN1; ++i) {
      far_spectrum = farend_fft[0][i] * farend_fft[0][i] +
                     farend_fft[1][i] * farend_fft[1][i];
      // Calculate the magnitude spectrum.
      abs_far_spectrum[i] = sqrtf(far_spectrum);
    }
    RegressorPower(aec->num_partitions, aec->xfBufBlockPos, aec->xfBuf,
                   aec->xPow);
  } else {
    for (i = 0; i < PART_LEN1; ++i) {
      far_spectrum = farend_fft[0][i] * farend_fft[0][i] +
                     farend_fft[1][i] * farend_fft[1][i];
      aec->xPow[i] =
          gPow[0] * aec->xPow[i] + gPow[1] * aec->num_partitions * far_spectrum;
      // Calculate the magnitude spectrum.
      abs_far_spectrum[i] = sqrtf(far_spectrum);
    }
  }

  for (i = 0; i < PART_LEN1; ++i) {
    near_spectrum = nearend_fft[0][i] * nearend_fft[0][i] +
                    nearend_fft[1][i] * nearend_fft[1][i];
    aec->dPow[i] = gPow[0] * aec->dPow[i] + gPow[1] * near_spectrum;
    // Calculate the magnitude spectrum.
    abs_near_spectrum[i] = sqrtf(near_spectrum);
  }

  // Estimate noise power. Wait until dPow is more stable.
  if (aec->noiseEstCtr > 50) {
    for (i = 0; i < PART_LEN1; i++) {
      if (aec->dPow[i] < aec->dMinPow[i]) {
        aec->dMinPow[i] =
            (aec->dPow[i] + step * (aec->dMinPow[i] - aec->dPow[i])) * ramp;
      } else {
        aec->dMinPow[i] *= ramp;
      }
    }
  }

  // Smooth increasing noise power from zero at the start,
  // to avoid a sudden burst of comfort noise.
  if (aec->noiseEstCtr < noiseInitBlocks) {
    aec->noiseEstCtr++;
    for (i = 0; i < PART_LEN1; i++) {
      if (aec->dMinPow[i] > aec->dInitMinPow[i]) {
        aec->dInitMinPow[i] = gInitNoise[0] * aec->dInitMinPow[i] +
                              gInitNoise[1] * aec->dMinPow[i];
      } else {
        aec->dInitMinPow[i] = aec->dMinPow[i];
      }
    }
    aec->noisePow = aec->dInitMinPow;
  } else {
    aec->noisePow = aec->dMinPow;
  }

  // Block wise delay estimation used for logging
  if (aec->delay_logging_enabled) {
    if (WebRtc_AddFarSpectrumFloat(aec->delay_estimator_farend,
                                   abs_far_spectrum, PART_LEN1) == 0) {
      int delay_estimate = WebRtc_DelayEstimatorProcessFloat(
          aec->delay_estimator, abs_near_spectrum, PART_LEN1);
      if (delay_estimate >= 0) {
        // Update delay estimate buffer.
        aec->delay_histogram[delay_estimate]++;
        aec->num_delay_values++;
      }
      if (aec->delay_metrics_delivered == 1 &&
          aec->num_delay_values >= kDelayMetricsAggregationWindow) {
        UpdateDelayMetrics(aec);
      }
    }
  }
#ifdef AEC_ROCORD
  short temp_near_in[PART_LEN] = {0};
  short temp_far_in[PART_LEN] = {0};
  short temp_out[PART_LEN] = {0};

  for (i = 0; i < PART_LEN; i++) {
    if (farend_extended_block_lowest_band[i + PART_LEN] > 32767) {
      temp_far_in[i] = 32767;
    } else if (farend_extended_block_lowest_band[i + PART_LEN] < -32768) {
      temp_far_in[i] = -32768;
    } else {
      temp_far_in[i] = (short)(farend_extended_block_lowest_band[i + PART_LEN]);
    }

    if (nearend_block[0][i] > 32767) {
      temp_near_in[i] = 32767;
    } else if (nearend_block[0][i] < -32768) {
      temp_near_in[i] = -32768;
    } else {
      temp_near_in[i] = (short)(nearend_block[0][i]);
    }
  }
  if (aec->f_aec_near_in != NULL) {
    fwrite(temp_near_in, 2, PART_LEN, aec->f_aec_near_in);
  }
  if (aec->f_aec_far_in != NULL) {
    fwrite(temp_far_in, 2, PART_LEN, aec->f_aec_far_in);
  }
#endif
#ifdef AUMDF_FILTER
  ProcessNearendBlockMdf(aec, aec->num_partitions, &aec->xfBufBlockPos,
                         farend_extended_block_lowest_band, nearend_block,
                         output_block);
#else
  // Perform echo subtraction.
  // printf("aec->xfBufBlockPos = %d, aec->num_partitions = %d\n",
  // aec->xfBufBlockPos, aec->num_partitions);
  EchoSubtraction(
      aec->ooura_fft, aec->num_partitions, aec->extended_filter_enabled,
      &aec->extreme_filter_divergence, aec->filter_step_size,
      aec->error_threshold, &farend_fft[0][0], &aec->xfBufBlockPos, aec->xfBuf,
      &nearend_block[0][0], aec->xPow, aec->wfBuf, echo_subtractor_output);
  aec->data_dumper->DumpRaw("aec_h_fft", PART_LEN1 * aec->num_partitions,
                            &aec->wfBuf[0][0]);
  aec->data_dumper->DumpRaw("aec_h_fft", PART_LEN1 * aec->num_partitions,
                            &aec->wfBuf[1][0]);

  aec->data_dumper->DumpWav("aec_out_linear", PART_LEN, echo_subtractor_output,
                            std::min(aec->sampFreq, 16000), 1);

  if (aec->metricsMode == 1) {
    UpdateLevel(&aec->linoutlevel,
                CalculatePower(echo_subtractor_output, PART_LEN));
  }

  // Perform echo suppression.
  EchoSuppression(aec->ooura_fft, aec, nearend_extended_block_lowest_band,
                  farend_extended_block_lowest_band, echo_subtractor_output,
                  output_block);
#endif
#ifdef AEC_ROCORD
  for (i = 0; i < PART_LEN; i++) {
    if (output_block[0][i] > 32767) {
      temp_out[i] = 32767;
    } else if (output_block[0][i] < -32768) {
      temp_out[i] = -32768;
    } else {
      temp_out[i] = (short)(output_block[0][i]);
    }
  }
  if (aec->f_aec_near_out != NULL) {
    fwrite(temp_out, 2, PART_LEN, aec->f_aec_near_out);
  }
#endif
  if (aec->metricsMode == 1) {
    UpdateLevel(&aec->nlpoutlevel,
                CalculatePower(&output_block[0][0], PART_LEN));
    UpdateMetrics(aec);
  }
#ifndef AUMDF_FILTER
  // Store the nearend signal until the next frame.
  for (i = 0; i < aec->num_bands; ++i) {
    memcpy(&aec->previous_nearend_block[i][0], &nearend_block[i][0],
           sizeof(float) * PART_LEN);
  }
#endif

  aec->data_dumper->DumpWav("aec_out", PART_LEN, &output_block[0][0],
                            std::min(aec->sampFreq, 16000), 1);
}

AecCore* WebRtcAec_CreateAec(int instance_count) {
  AecCore* aec = new AecCore(instance_count);

  if (!aec) {
    return NULL;
  }
  aec->nearend_buffer_size = 0;
  memset(&aec->nearend_buffer[0], 0, sizeof(aec->nearend_buffer));
  // Start the output buffer with zeros to be able to produce
  // a full output frame in the first frame.
  aec->output_buffer_size = PART_LEN - (FRAME_LEN - PART_LEN);
  memset(&aec->output_buffer[0], 0, sizeof(aec->output_buffer));

  aec->delay_estimator_farend =
      WebRtc_CreateDelayEstimatorFarend(PART_LEN1, kHistorySizeBlocks);
  if (aec->delay_estimator_farend == NULL) {
    WebRtcAec_FreeAec(aec);
    return NULL;
  }
  // We create the delay_estimator with the same amount of maximum lookahead as
  // the delay history size (kHistorySizeBlocks) for symmetry reasons.
  aec->delay_estimator = WebRtc_CreateDelayEstimator(
      aec->delay_estimator_farend, kHistorySizeBlocks);
  if (aec->delay_estimator == NULL) {
    WebRtcAec_FreeAec(aec);
    return NULL;
  }
#ifdef WEBRTC_ANDROID
  aec->delay_agnostic_enabled = 1;  // DA-AEC enabled by default.
  // DA-AEC assumes the system is causal from the beginning and will self adjust
  // the lookahead when shifting is required.
  WebRtc_set_lookahead(aec->delay_estimator, 0);
#else
  aec->delay_agnostic_enabled = 0;
  WebRtc_set_lookahead(aec->delay_estimator, kLookaheadBlocks);
#endif
  aec->extended_filter_enabled = 0;
  aec->refined_adaptive_filter_enabled = false;

  // Assembly optimization
#ifndef AUMDF_FILTER
  WebRtcAec_FilterFar = FilterFar;
  WebRtcAec_ScaleErrorSignal = ScaleErrorSignal;
  WebRtcAec_FilterAdaptation = FilterAdaptation;
#endif
  WebRtcAec_Overdrive = Overdrive;
  WebRtcAec_Suppress = Suppress;
  WebRtcAec_ComputeCoherence = ComputeCoherence;
  WebRtcAec_UpdateCoherenceSpectra = UpdateCoherenceSpectra;
  WebRtcAec_StoreAsComplex = StoreAsComplex;
#ifndef AUMDF_FILTER
  WebRtcAec_PartitionDelay = PartitionDelay;
#endif
  WebRtcAec_WindowData = WindowData;

#if defined(WEBRTC_ARCH_X86_FAMILY)
  if (WebRtc_GetCPUInfo(kSSE2)) {
    WebRtcAec_InitAec_SSE2();
  }
#endif

#if defined(MIPS_FPU_LE)
  WebRtcAec_InitAec_mips();
#endif

#if defined(WEBRTC_HAS_NEON)
  WebRtcAec_InitAec_neon();
#endif
#ifdef AEC_ROCORD
  aec->f_aec_near_in = NULL;
  aec->f_aec_far_in = NULL;
  aec->f_aec_near_out = NULL;
  aec->f_aec_linear_out = NULL;
  aec->f_far_in = NULL;
//  aec->f_far_in2 = NULL;
#endif
  return aec;
}

void WebRtcAec_FreeAec(AecCore* aec) {
  if (aec == NULL) {
    return;
  }

  WebRtc_FreeDelayEstimator(aec->delay_estimator);
  WebRtc_FreeDelayEstimatorFarend(aec->delay_estimator_farend);
#ifdef AEC_ROCORD
  if (aec->f_aec_near_in != NULL) {
    fclose(aec->f_aec_near_in);
    aec->f_aec_near_in = NULL;
  }
  if (aec->f_aec_far_in != NULL) {
    fclose(aec->f_aec_far_in);
    aec->f_aec_far_in = NULL;
  }
  if (aec->f_aec_near_out != NULL) {
    fclose(aec->f_aec_near_out);
    aec->f_aec_near_out = NULL;
  }
  if (aec->f_aec_linear_out != NULL) {
    fclose(aec->f_aec_linear_out);
    aec->f_aec_linear_out = NULL;
  }
  if (aec->f_far_in != NULL) {
    fclose(aec->f_far_in);
    aec->f_far_in = NULL;
  }
    /* if (aec->f_far_in2 != NULL)
     {
             fclose(aec->f_far_in2);
             aec->f_far_in2 = NULL;
     }*/
#endif
#ifdef AUMDF_FILTER
  linear_echo_state_destroy(aec->st);
#endif
  delete aec;
}

static void SetAdaptiveFilterStepSize(AecCore* aec) {
  // Extended filter adaptation parameter.
  // TODO(ajm): No narrowband tuning yet.
  const float kExtendedMu = 0.4f;

  if (aec->refined_adaptive_filter_enabled) {
    aec->filter_step_size = 0.05f;
  } else {
    if (aec->extended_filter_enabled) {
      aec->filter_step_size = kExtendedMu;
    } else {
      if (aec->sampFreq == 8000) {
        aec->filter_step_size = 0.6f;
      } else {
        aec->filter_step_size = 0.5f;
      }
    }
  }
}

static void SetErrorThreshold(AecCore* aec) {
  // Extended filter adaptation parameter.
  // TODO(ajm): No narrowband tuning yet.
  static const float kExtendedErrorThreshold = 1.0e-6f;

  if (aec->extended_filter_enabled) {
    aec->error_threshold = kExtendedErrorThreshold;
  } else {
    if (aec->sampFreq == 8000) {
      aec->error_threshold = 2e-6f;
    } else {
      aec->error_threshold = 1.5e-6f;
    }
  }
}

int WebRtcAec_InitAec(AecCore* aec, int sampFreq) {
  int i;
  aec->data_dumper->InitiateNewSetOfRecordings();

  aec->sampFreq = sampFreq;

  SetAdaptiveFilterStepSize(aec);
  SetErrorThreshold(aec);

  if (sampFreq == 8000) {
    aec->num_bands = 1;
  } else {
    aec->num_bands = (size_t)(sampFreq / 16000);
  }

  // Start the output buffer with zeros to be able to produce
  // a full output frame in the first frame.
  aec->output_buffer_size = PART_LEN - (FRAME_LEN - PART_LEN);
  memset(&aec->output_buffer[0], 0, sizeof(aec->output_buffer));
  aec->nearend_buffer_size = 0;
  memset(&aec->nearend_buffer[0], 0, sizeof(aec->nearend_buffer));

  // Initialize far-end buffer.
  aec->farend_block_buffer_.ReInit();

  aec->system_delay = 0;

  if (WebRtc_InitDelayEstimatorFarend(aec->delay_estimator_farend) != 0) {
    return -1;
  }
  if (WebRtc_InitDelayEstimator(aec->delay_estimator) != 0) {
    return -1;
  }
  aec->delay_logging_enabled = 0;
  aec->delay_metrics_delivered = 0;
  memset(aec->delay_histogram, 0, sizeof(aec->delay_histogram));
  aec->num_delay_values = 0;
  aec->delay_median = -1;
  aec->delay_std = -1;
  aec->fraction_poor_delays = -1.0f;

  aec->previous_delay = -2;  // (-2): Uninitialized.
  aec->delay_correction_count = 0;
  aec->shift_offset = kInitialShiftOffset;
  aec->delay_quality_threshold = kDelayQualityThresholdMin;

  aec->num_partitions = kNormalNumPartitions;

  // Update the delay estimator with filter length.  We use half the
  // |num_partitions| to take the echo path into account.  In practice we say
  // that the echo has a duration of maximum half |num_partitions|, which is not
  // true, but serves as a crude measure.
  WebRtc_set_allowed_offset(aec->delay_estimator, aec->num_partitions / 2);
  // TODO(bjornv): I currently hard coded the enable.  Once we've established
  // that AECM has no performance regression, robust_validation will be enabled
  // all the time and the APIs to turn it on/off will be removed.  Hence, remove
  // this line then.
  WebRtc_enable_robust_validation(aec->delay_estimator, 1);
  aec->frame_count = 0;

  // Default target suppression mode.
  aec->nlp_mode = 1;

  // Sampling frequency multiplier w.r.t. 8 kHz.
  // In case of multiple bands we process the lower band in 16 kHz, hence the
  // multiplier is always 2.
  if (aec->num_bands > 1) {
    aec->mult = 2;
  } else {
    aec->mult = static_cast<int16_t>(aec->sampFreq) / 8000;
  }

  aec->farBufWritePos = 0;
  aec->farBufReadPos = 0;

  aec->inSamples = 0;
  aec->outSamples = 0;
  aec->knownDelay = 0;

  // Initialize buffers
  memset(aec->previous_nearend_block, 0, sizeof(aec->previous_nearend_block));
  memset(aec->eBuf, 0, sizeof(aec->eBuf));

  memset(aec->xPow, 0, sizeof(aec->xPow));
  memset(aec->dPow, 0, sizeof(aec->dPow));
  memset(aec->dInitMinPow, 0, sizeof(aec->dInitMinPow));
  aec->noisePow = aec->dInitMinPow;
  aec->noiseEstCtr = 0;

  // Initial comfort noise power
  for (i = 0; i < PART_LEN1; i++) {
    aec->dMinPow[i] = 1.0e6f;
  }

  // Holds the last block written to
  aec->xfBufBlockPos = 0;
  // TODO(peah): Investigate need for these initializations. Deleting them
  // doesn't change the output at all and yields 0.4% overall speedup.
  memset(aec->xfBuf, 0, sizeof(complex_t) * kExtendedNumPartitions * PART_LEN1);
  memset(aec->wfBuf, 0, sizeof(complex_t) * kExtendedNumPartitions * PART_LEN1);
  memset(aec->coherence_state.sde, 0, sizeof(complex_t) * PART_LEN1);
  memset(aec->coherence_state.sxd, 0, sizeof(complex_t) * PART_LEN1);
  memset(aec->xfwBuf, 0,
         sizeof(complex_t) * kExtendedNumPartitions * PART_LEN1);
  memset(aec->coherence_state.se, 0, sizeof(float) * PART_LEN1);

  // To prevent numerical instability in the first block.
  for (i = 0; i < PART_LEN1; i++) {
    aec->coherence_state.sd[i] = 1;
  }
  for (i = 0; i < PART_LEN1; i++) {
    aec->coherence_state.sx[i] = 1;
  }

  memset(aec->hNs, 0, sizeof(aec->hNs));
  memset(aec->outBuf, 0, sizeof(float) * PART_LEN);

  aec->hNlFbMin = 1;
  aec->hNlFbLocalMin = 1;
  aec->hNlXdAvgMin = 1;
  aec->hNlNewMin = 0;
  aec->hNlMinCtr = 0;
  aec->overDrive = 2;
  aec->overdrive_scaling = 2;
  aec->delayIdx = 0;
  aec->stNearState = 0;
  aec->echoState = 0;
  aec->divergeState = 0;

  aec->seed = 777;
  aec->delayEstCtr = 0;

  aec->extreme_filter_divergence = 0;

  // Metrics disabled by default
  aec->metricsMode = 0;
  InitMetrics(aec);
#ifdef AEC_ROCORD
  char current_path[MAX_PATH];
  bool savePcm = false;
  DWORD len = GetModuleFileNameA(NULL, current_path, MAX_PATH);
  if (len > 0) {
    strrchr(current_path, '\\')[0] = '\0';
    char cfgFile[256];
    sprintf(cfgFile, "%s\\savepcm.txt", current_path);
    FILE* fp = fopen(cfgFile, "rb");
    if (fp != nullptr) {
      fclose(fp);
      savePcm = true;
    }
  }

  if (savePcm) {
    if (aec->f_aec_near_in == NULL) {
      char filename[256];
      if (strlen(current_path) > 0)
        sprintf(filename, "%s\\aec_near_in_%p.pcm", current_path, aec);
      else
        sprintf(filename, "aec_near_in_%p.pcm", aec);
      aec->f_aec_near_in = fopen(filename, "wb");
    }
    if (aec->f_aec_far_in == NULL) {
      char filename[256];
      if (strlen(current_path) > 0)
        sprintf(filename, "%s\\aec_far_in_%p.pcm", current_path, aec);
      else
        sprintf(filename, "aec_far_in_%p.pcm", aec);
      aec->f_aec_far_in = fopen(filename, "wb");
    }
    if (aec->f_aec_near_out == NULL) {
      char filename[256];
      if (strlen(current_path) > 0)
        sprintf(filename, "%s\\aec_out_%p.pcm", current_path, aec);
      else
        sprintf(filename, "aec_out_%p.pcm", aec);
      aec->f_aec_near_out = fopen(filename, "wb");
    }
  }

    // if (aec->f_aec_linear_out == NULL)
    //{
    // aec->f_aec_linear_out = fopen("aec_linear_out.pcm", "wb");
    //}
    // if (aec->f_far_in == NULL)
    //{
    // aec->f_far_in = fopen("far_in1.pcm", "wb");
    //}
    /* if (aec->f_far_in2 == NULL)
     {
             aec->f_far_in2 = fopen("far_in2.pcm", "wb");
     }*/
#endif
#ifdef AUMDF_FILTER
  memset(aec->near_buf, 0, sizeof(aec->near_buf));
  int samplerate = 16000;
  if (aec->sampFreq > 16000) {
    samplerate = 16000;
  } else {
    samplerate = aec->sampFreq;
  }
  aec->st = linear_echo_state_init(PART_LEN, TAILLENGTH, samplerate);
#endif
  return 0;
}

void WebRtcAec_BufferFarendBlock(AecCore* aec, const float* farend) {
  // Check if the buffer is full, and in that case flush the oldest data.
  if (aec->farend_block_buffer_.AvaliableSpace() < 1) {
    aec->farend_block_buffer_.AdjustSize(1);
  }
  aec->farend_block_buffer_.Insert(farend);
}

int WebRtcAec_AdjustFarendBufferSizeAndSystemDelay(AecCore* aec,
                                                   int buffer_size_decrease) {
  int achieved_buffer_size_decrease =
      aec->farend_block_buffer_.AdjustSize(buffer_size_decrease);
  aec->system_delay -= achieved_buffer_size_decrease * PART_LEN;
  return achieved_buffer_size_decrease;
}

void FormNearendBlock(
    size_t nearend_start_index,
    size_t num_bands,
    const float* const* nearend_frame,
    size_t num_samples_from_nearend_frame,
    const float nearend_buffer[NUM_HIGH_BANDS_MAX + 1]
                              [PART_LEN - (FRAME_LEN - PART_LEN)],
    float nearend_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN]) {
  RTC_DCHECK_LE(num_samples_from_nearend_frame, PART_LEN);
  const int num_samples_from_buffer = PART_LEN - num_samples_from_nearend_frame;

  if (num_samples_from_buffer > 0) {
    for (size_t i = 0; i < num_bands; ++i) {
      memcpy(&nearend_block[i][0], &nearend_buffer[i][0],
             num_samples_from_buffer * sizeof(float));
    }
  }

  for (size_t i = 0; i < num_bands; ++i) {
    memcpy(&nearend_block[i][num_samples_from_buffer],
           &nearend_frame[i][nearend_start_index],
           num_samples_from_nearend_frame * sizeof(float));
  }
}

void BufferNearendFrame(
    size_t nearend_start_index,
    size_t num_bands,
    const float* const* nearend_frame,
    size_t num_samples_to_buffer,
    float nearend_buffer[NUM_HIGH_BANDS_MAX + 1]
                        [PART_LEN - (FRAME_LEN - PART_LEN)]) {
  for (size_t i = 0; i < num_bands; ++i) {
    memcpy(&nearend_buffer[i][0],
           &nearend_frame[i][nearend_start_index + FRAME_LEN -
                             num_samples_to_buffer],
           num_samples_to_buffer * sizeof(float));
  }
}

void BufferOutputBlock(
    size_t num_bands,
    const float output_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN],
    size_t* output_buffer_size,
    float output_buffer[NUM_HIGH_BANDS_MAX + 1][2 * PART_LEN]) {
  for (size_t i = 0; i < num_bands; ++i) {
    memcpy(&output_buffer[i][*output_buffer_size], &output_block[i][0],
           PART_LEN * sizeof(float));
  }
  (*output_buffer_size) += PART_LEN;
}

void FormOutputFrame(size_t output_start_index,
                     size_t num_bands,
                     size_t* output_buffer_size,
                     float output_buffer[NUM_HIGH_BANDS_MAX + 1][2 * PART_LEN],
                     float* const* output_frame) {
  RTC_DCHECK_LE(FRAME_LEN, *output_buffer_size);
  for (size_t i = 0; i < num_bands; ++i) {
    memcpy(&output_frame[i][output_start_index], &output_buffer[i][0],
           FRAME_LEN * sizeof(float));
  }
  (*output_buffer_size) -= FRAME_LEN;
  if (*output_buffer_size > 0) {
    RTC_DCHECK_GE(2 * PART_LEN - FRAME_LEN, (*output_buffer_size));
    for (size_t i = 0; i < num_bands; ++i) {
      memcpy(&output_buffer[i][0], &output_buffer[i][FRAME_LEN],
             (*output_buffer_size) * sizeof(float));
    }
  }
}

void WebRtcAec_ProcessFrames(AecCore* aec,
                             const float* const* nearend,
                             size_t num_bands,
                             size_t num_samples,
                             int knownDelay,
                             float* const* out) {
  RTC_DCHECK(num_samples == 80 || num_samples == 160);
  aec->frame_count++;
  // For each frame the process is as follows:
  // 1) If the system_delay indicates on being too small for processing a
  //    frame we stuff the buffer with enough data for 10 ms.
  // 2 a) Adjust the buffer to the system delay, by moving the read pointer.
  //   b) Apply signal based delay correction, if we have detected poor AEC
  //    performance.
  // 3) TODO(bjornv): Investigate if we need to add this:
  //    If we can't move read pointer due to buffer size limitations we
  //    flush/stuff the buffer.
  // 4) Process as many partitions as possible.
  // 5) Update the |system_delay| with respect to a full frame of FRAME_LEN
  //    samples. Even though we will have data left to process (we work with
  //    partitions) we consider updating a whole frame, since that's the
  //    amount of data we input and output in audio_processing.
  // 6) Update the outputs.

  // The AEC has two different delay estimation algorithms built in.  The
  // first relies on delay input values from the user and the amount of
  // shifted buffer elements is controlled by |knownDelay|.  This delay will
  // give a guess on how much we need to shift far-end buffers to align with
  // the near-end signal.  The other delay estimation algorithm uses the
  // far- and near-end signals to find the offset between them.  This one
  // (called "signal delay") is then used to fine tune the alignment, or
  // simply compensate for errors in the system based one.
  // Note that the two algorithms operate independently.  Currently, we only
  // allow one algorithm to be turned on.

  RTC_DCHECK_EQ(aec->num_bands, num_bands);

  for (size_t j = 0; j < num_samples; j += FRAME_LEN) {
    // 1) At most we process |aec->mult|+1 partitions in 10 ms. Make sure we
    // have enough far-end data for that by stuffing the buffer if the
    // |system_delay| indicates others.
    if (aec->system_delay < FRAME_LEN) {
      // We don't have enough data so we rewind 10 ms.
      WebRtcAec_AdjustFarendBufferSizeAndSystemDelay(aec, -(aec->mult + 1));
    }

    if (!aec->delay_agnostic_enabled) {
      // 2 a) Compensate for a possible change in the system delay.

      // TODO(bjornv): Investigate how we should round the delay difference;
      // right now we know that incoming |knownDelay| is underestimated when
      // it's less than |aec->knownDelay|. We therefore, round (-32) in that
      // direction. In the other direction, we don't have this situation, but
      // might flush one partition too little. This can cause non-causality,
      // which should be investigated. Maybe, allow for a non-symmetric
      // rounding, like -16.
      int move_elements = (aec->knownDelay - knownDelay - 32) / PART_LEN;
      int moved_elements = aec->farend_block_buffer_.AdjustSize(move_elements);
      MaybeLogDelayAdjustment(moved_elements * (aec->sampFreq == 8000 ? 8 : 4),
                              DelaySource::kSystemDelay);
      aec->knownDelay -= moved_elements * PART_LEN;
    } else {
      // 2 b) Apply signal based delay correction.
      int move_elements = SignalBasedDelayCorrection(aec);
      int moved_elements = aec->farend_block_buffer_.AdjustSize(move_elements);
      MaybeLogDelayAdjustment(moved_elements * (aec->sampFreq == 8000 ? 8 : 4),
                              DelaySource::kDelayAgnostic);
      int far_near_buffer_diff =
          aec->farend_block_buffer_.Size() -
          (aec->nearend_buffer_size + FRAME_LEN) / PART_LEN;
      WebRtc_SoftResetDelayEstimator(aec->delay_estimator, moved_elements);
      WebRtc_SoftResetDelayEstimatorFarend(aec->delay_estimator_farend,
                                           moved_elements);
      // If we rely on reported system delay values only, a buffer underrun here
      // can never occur since we've taken care of that in 1) above.  Here, we
      // apply signal based delay correction and can therefore end up with
      // buffer underruns since the delay estimation can be wrong.  We therefore
      // stuff the buffer with enough elements if needed.
      if (far_near_buffer_diff < 0) {
        WebRtcAec_AdjustFarendBufferSizeAndSystemDelay(aec,
                                                       far_near_buffer_diff);
      }
    }

    static_assert(
        16 == (FRAME_LEN - PART_LEN),
        "These constants need to be properly related for this code to work");
    float output_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN];
    float nearend_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN];
    float farend_extended_block_lowest_band[PART_LEN2];
    // Form and process a block of nearend samples, buffer the output block of
    // samples.
    aec->farend_block_buffer_.ExtractExtendedBlock(
        farend_extended_block_lowest_band);
    FormNearendBlock(j, num_bands, nearend, PART_LEN - aec->nearend_buffer_size,
                     aec->nearend_buffer, nearend_block);
    ProcessNearendBlock(aec, farend_extended_block_lowest_band, nearend_block,
                        output_block);
    BufferOutputBlock(num_bands, output_block, &aec->output_buffer_size,
                      aec->output_buffer);

    if ((FRAME_LEN - PART_LEN + aec->nearend_buffer_size) == PART_LEN) {
      // When possible (every fourth frame) form and process a second block of
      // nearend samples, buffer the output block of samples.
      aec->farend_block_buffer_.ExtractExtendedBlock(
          farend_extended_block_lowest_band);
      FormNearendBlock(j + FRAME_LEN - PART_LEN, num_bands, nearend, PART_LEN,
                       aec->nearend_buffer, nearend_block);
      ProcessNearendBlock(aec, farend_extended_block_lowest_band, nearend_block,
                          output_block);
      BufferOutputBlock(num_bands, output_block, &aec->output_buffer_size,
                        aec->output_buffer);

      // Reset the buffer size as there are no samples left in the nearend input
      // to buffer.
      aec->nearend_buffer_size = 0;
    } else {
      // Buffer the remaining samples in the nearend input.
      aec->nearend_buffer_size += FRAME_LEN - PART_LEN;
      BufferNearendFrame(j, num_bands, nearend, aec->nearend_buffer_size,
                         aec->nearend_buffer);
    }

    // 5) Update system delay with respect to the entire frame.
    aec->system_delay -= FRAME_LEN;

    // 6) Form the output frame.
    FormOutputFrame(j, num_bands, &aec->output_buffer_size, aec->output_buffer,
                    out);
  }
}

int WebRtcAec_GetDelayMetricsCore(AecCore* self,
                                  int* median,
                                  int* std,
                                  float* fraction_poor_delays) {
  RTC_DCHECK(self);
  RTC_DCHECK(median);
  RTC_DCHECK(std);

  if (self->delay_logging_enabled == 0) {
    // Logging disabled.
    return -1;
  }

  if (self->delay_metrics_delivered == 0) {
    UpdateDelayMetrics(self);
    self->delay_metrics_delivered = 1;
  }
  *median = self->delay_median;
  *std = self->delay_std;
  *fraction_poor_delays = self->fraction_poor_delays;

  return 0;
}

int WebRtcAec_echo_state(AecCore* self) {
  return self->echoState;
}

void WebRtcAec_GetEchoStats(AecCore* self,
                            Stats* erl,
                            Stats* erle,
                            Stats* a_nlp,
                            float* divergent_filter_fraction) {
  RTC_DCHECK(erl);
  RTC_DCHECK(erle);
  RTC_DCHECK(a_nlp);
  *erl = self->erl;
  *erle = self->erle;
  *a_nlp = self->aNlp;
  *divergent_filter_fraction =
      self->divergent_filter_fraction.GetLatestFraction();
}

void WebRtcAec_SetConfigCore(AecCore* self,
                             int nlp_mode,
                             int metrics_mode,
                             int delay_logging) {
  RTC_DCHECK_GE(nlp_mode, 0);
  RTC_DCHECK_LT(nlp_mode, 4);
  if(nlp_mode == 3)
  {
  	self->nlp_mode = nlp_mode;
  }
  else
  {
  	self->nlp_mode = 1;
  }
  self->metricsMode = metrics_mode;
  if (self->metricsMode) {
    InitMetrics(self);
  }
  // Turn on delay logging if it is either set explicitly or if delay agnostic
  // AEC is enabled (which requires delay estimates).
  self->delay_logging_enabled = delay_logging || self->delay_agnostic_enabled;
  if (self->delay_logging_enabled) {
    memset(self->delay_histogram, 0, sizeof(self->delay_histogram));
  }
}

void WebRtcAec_enable_delay_agnostic(AecCore* self, int enable) {
  self->delay_agnostic_enabled = 0;
}

int WebRtcAec_delay_agnostic_enabled(AecCore* self) {
  return self->delay_agnostic_enabled;
}

void WebRtcAec_enable_refined_adaptive_filter(AecCore* self, bool enable) {
  self->refined_adaptive_filter_enabled = enable;
  SetAdaptiveFilterStepSize(self);
  SetErrorThreshold(self);
}

bool WebRtcAec_refined_adaptive_filter_enabled(const AecCore* self) {
  return self->refined_adaptive_filter_enabled;
}

void WebRtcAec_enable_extended_filter(AecCore* self, int enable) {
  enable = 0;
  self->extended_filter_enabled = enable;
  SetAdaptiveFilterStepSize(self);
  SetErrorThreshold(self);
  self->num_partitions = enable ? kExtendedNumPartitions : kNormalNumPartitions;
  // Update the delay estimator with filter length.  See InitAEC() for details.
  WebRtc_set_allowed_offset(self->delay_estimator, self->num_partitions / 2);
}

int WebRtcAec_extended_filter_enabled(AecCore* self) {
  return self->extended_filter_enabled;
}

int WebRtcAec_system_delay(AecCore* self) {
  return self->system_delay;
}

void WebRtcAec_SetSystemDelay(AecCore* self, int delay) {
  RTC_DCHECK_GE(delay, 0);
  self->system_delay = delay;
}
}  // namespace webrtc
