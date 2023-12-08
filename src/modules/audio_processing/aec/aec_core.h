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
 * Specifies the interface for the AEC core.
 */

#ifndef MODULES_AUDIO_PROCESSING_AEC_AEC_CORE_H_
#define MODULES_AUDIO_PROCESSING_AEC_AEC_CORE_H_

#include <stddef.h>

#include <memory>

#ifndef AUMDF_FILTER
#define AUMDF_FILTER
#endif

#ifndef AEC_ROCORD
#define AEC_ROCORD
#endif

#ifndef LINEAR_FILTER_ROCORD
//#define LINEAR_FILTER_ROCORD
#endif

extern "C" {
#include "common_audio/ring_buffer.h"
}
#include "modules/audio_processing/aec/aec_common.h"
#include "modules/audio_processing/utility/block_mean_calculator.h"
#include "modules/audio_processing/utility/ooura_fft.h"
#include "rtc_base/constructor_magic.h"

namespace webrtc {
#define FRAME_LEN 80
#define PART_LEN 64               // Length of partition
#define PART_LEN1 (PART_LEN + 1)  // Unique fft coefficients
#define PART_LEN2 (PART_LEN * 2)  // Length of partition * 2
#define NUM_HIGH_BANDS_MAX 2      // Max number of high bands
//#define NOMINMAX
class ApmDataDumper;

typedef float complex_t[2];
// For performance reasons, some arrays of complex numbers are replaced by twice
// as long arrays of float, all the real parts followed by all the imaginary
// ones (complex_t[SIZE] -> float[2][SIZE]). This allows SIMD optimizations and
// is better than two arrays (one for the real parts and one for the imaginary
// parts) as this other way would require two pointers instead of one and cause
// extra register spilling. This also allows the offsets to be calculated at
// compile time.

// Metrics
enum { kOffsetLevel = -100 };

typedef struct Stats {
  float instant;
  float average;
  float min;
  float max;
  float sum;
  float hisum;
  float himean;
  size_t counter;
  size_t hicounter;
} Stats;


// Delay estimator constants, used for logging and delay compensation if
// if reported delays are disabled.
enum { kLookaheadBlocks = 15 };
enum {
  // 500 ms for 16 kHz which is equivalent with the limit of reported delays.
  kHistorySizeBlocks = 125
};

typedef struct PowerLevel {
  PowerLevel();

  BlockMeanCalculator framelevel;
  BlockMeanCalculator averagelevel;
  float minlevel;
} PowerLevel;

class BlockBuffer {
 public:
  BlockBuffer();
  ~BlockBuffer();
  void ReInit();
  void Insert(const float block[PART_LEN]);
  void ExtractExtendedBlock(float extended_block[PART_LEN]);
  int AdjustSize(int buffer_size_decrease);
  size_t Size();
  size_t AvaliableSpace();

 private:
  RingBuffer* buffer_;
};

class DivergentFilterFraction {
 public:
  DivergentFilterFraction();

  // Reset.
  void Reset();

  void AddObservation(const PowerLevel& nearlevel,
                      const PowerLevel& linoutlevel,
                      const PowerLevel& nlpoutlevel);

  // Return the latest fraction.
  float GetLatestFraction() const;

 private:
  // Clear all values added.
  void Clear();

  size_t count_;
  size_t occurrence_;
  float fraction_;

  RTC_DISALLOW_COPY_AND_ASSIGN(DivergentFilterFraction);
};

typedef struct CoherenceState {
  complex_t sde[PART_LEN1];  // cross-psd of nearend and error
  complex_t sxd[PART_LEN1];  // cross-psd of farend and nearend
  float sx[PART_LEN1], sd[PART_LEN1], se[PART_LEN1];  // far, near, error psd
} CoherenceState;

#ifndef AUMDF_FILTER
// Number of partitions for the extended filter mode. The first one is an enum
// to be used in array declarations, as it represents the maximum filter length.
enum { kExtendedNumPartitions = 32 };
static const int kNormalNumPartitions = 12;
#else
#define COUNTER 6
#define TAILLENGTH 4096
#define MM ((TAILLENGTH+63)/64)
#define SAMPLERATE 16000
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Number of partitions for the extended filter mode. The first one is an enum
// to be used in array declarations, as it represents the maximum filter length.
enum { kExtendedNumPartitions = 64 };
static const int kNormalNumPartitions = TAILLENGTH / 64;

#define FLOAT_GT(a,b) ((a)>(b))
#define ABS32(x) ((x) < 0 ? (-(x)) : (x))    /**< Absolute 32-bit value.  */
#define MAX32(a,b) ((a) > (b) ? (a) : (b))   /**< Maximum 32-bit value.   */
#define NEG16(x) (-(x))
#define NEG32(x) (-(x))
#define EXTEND32(x) (x)
#define SHR32(a,shift) (a)
#define SHL32(a,shift) (a)
#define PSHR16(a,shift) (a)
#define PSHR32(a,shift) (a)
#define ADD32(a,b) ((a)+(b))
#define SUB32(a,b) ((a)-(b))
#define MULT16_16(a,b)     ((float)(a)*(float)(b))
#define MAC16_16(c,a,b)     ((c)+(float)(a)*(float)(b))

static const float MIN_LEAK = .005f;

/* Constants for the two-path filter */
static const float VAR1_SMOOTH = .36f;
static const float VAR2_SMOOTH = .7225f;
static const float VAR1_UPDATE = .5f;
static const float VAR2_UPDATE = .25f;
static const float VAR_BACKTRACK = 4.f;
#define TOP16(x) (x)

/** Speex echo cancellation state. */
struct LinearEchoState {
	int frame_size;           /**< Number of samples processed each time */
	int window_size;
	int M;
	int cancel_count;
	int adapted;
	int saturated;
	int screwed_up;
	int sampling_rate;
	float spec_average;
	float beta0;
	float beta_max;
	float sum_adapt;
	float leak_estimate;

	float e[PART_LEN2];      /* scratch */
	float x[PART_LEN2];      /* Far-end input buffer (2N) */
	float X[(MM + 1) * PART_LEN2];      /* Far-end buffer (M+1 frames) in frequency domain */
	float input[PART_LEN];  /* scratch */
	float y[PART_LEN2];      /* scratch */
	float last_y[PART_LEN2];
	float Y[PART_LEN2];      /* scratch */
	float E[PART_LEN2];
	float PHI[PART_LEN2];    /* scratch */
	float W[MM * PART_LEN2];      /* (Background) filter weights */
	float foreground[MM * PART_LEN2]; /* Foreground filter weights */
	float  Davg1;  /* 1st recursive average of the residual power difference */
	float  Davg2;  /* 2nd recursive average of the residual power difference */
	float   Dvar1;  /* Estimated variance of 1st estimator */
	float   Dvar2;  /* Estimated variance of 2nd estimator */

	float power[PART_LEN1];  /* Power of the far-end signal */
	float power_1[PART_LEN1];/* Inverse power of far-end */
	float wtmp[PART_LEN2];   /* scratch */
	float Rf[PART_LEN1];     /* scratch */
	float Yf[PART_LEN1];     /* scratch */
	float Xf[PART_LEN1];     /* scratch */
	float Eh[PART_LEN1];
	float Yh[PART_LEN1];
	float   Pey;
	float   Pyy;
	float prop[MM];
	void *fft_table;
	float memX, memD, memE;
	float memLinear;
	float preemph;
	float notch_radius;
	float notch_mem[2];
	short updateflag;
	short resetflag;
#ifdef LINEAR_FILTER_ROCORD
	FILE *f_linear_near_in;
	FILE *f_linear_far_in;
	FILE *f_linear_near_out;
	FILE *f_fft_in;
	FILE *f_fft_out;
	FILE *f_ifft_in;
	FILE *f_ifft_out;
	FILE *f_near_in;
	FILE *f_far_in;
#endif
};

LinearEchoState *linear_echo_state_init(int frame_size, int filter_length, int samplerate);

void linear_echo_state_destroy(LinearEchoState *st);

void linear_echo_cancellation(LinearEchoState *st, float *rec, float *play, short *out, short *linear_out, float *leakestimate, short *nlpFlag);

void linear_echo_state_reset(LinearEchoState *st);


static inline void *linear_alloc(int size)
{
	return calloc(size, 1);
}

static inline void linear_free(void *ptr)
{
	free(ptr);
}


#define S_MUL(a,b) ( (a)*(b) )
#define C_MUL(m,a,b) \
    do{ (m).r = (a).r*(b).r - (a).i*(b).i;\
        (m).i = (a).r*(b).i + (a).i*(b).r; }while(0)

#define C_MUL4(m,a,b) C_MUL(m,a,b)

#define C_FIXDIV(c,div) /* NOOP */
#define C_MULBYSCALAR( c, s ) \
    do{ (c).r *= (s);\
        (c).i *= (s); }while(0)

#ifndef CHECK_OVERFLOW_OP
#  define CHECK_OVERFLOW_OP(a,op,b) /* noop */
#endif

#define  C_ADD( res, a,b)\
    do { \
	    CHECK_OVERFLOW_OP((a).r,+,(b).r)\
	    CHECK_OVERFLOW_OP((a).i,+,(b).i)\
	    (res).r=(a).r+(b).r;  (res).i=(a).i+(b).i; \
    }while(0)

#define  C_SUB( res, a,b)\
    do { \
	    CHECK_OVERFLOW_OP((a).r,-,(b).r)\
	    CHECK_OVERFLOW_OP((a).i,-,(b).i)\
	    (res).r=(a).r-(b).r;  (res).i=(a).i-(b).i; \
    }while(0)

#define C_ADDTO( res , a)\
    do { \
	    CHECK_OVERFLOW_OP((res).r,+,(a).r)\
	    CHECK_OVERFLOW_OP((res).i,+,(a).i)\
	    (res).r += (a).r;  (res).i += (a).i;\
    }while(0)

#define C_SUBFROM( res , a)\
    do {\
	    CHECK_OVERFLOW_OP((res).r,-,(a).r)\
	    CHECK_OVERFLOW_OP((res).i,-,(a).i)\
	    (res).r -= (a).r;  (res).i -= (a).i; \
    }while(0)

#ifdef USE_SIMD
#  define KISS_FFT_COS(phase) _mm_set1_ps( cos(phase) )
#  define KISS_FFT_SIN(phase) _mm_set1_ps( sin(phase) )
#  define HALF_OF(x) ((x)*_mm_set1_ps(.5))
#else
#  define KISS_FFT_COS(phase) (float) cos(phase)
#  define KISS_FFT_SIN(phase) (float) sin(phase)
#  define HALF_OF(x) ((x)*.5)
#endif

#define  kf_cexp(x,phase) \
	do{ \
		(x)->r = KISS_FFT_COS(phase);\
		(x)->i = KISS_FFT_SIN(phase);\
	}while(0)
#define  kf_cexp2(x,phase) \
               do{ \
               (x)->r = spx_cos_norm((phase));\
               (x)->i = spx_cos_norm((phase)-32768);\
}while(0)

#ifdef USE_SIMD
# include <xmmintrin.h>
# define kiss_fft_scalar __m128
#define KISS_FFT_MALLOC(nbytes) memalign(16,nbytes)
#else	
# define kiss_fft_scalar float
#define KISS_FFT_MALLOC linear_alloc
#endif	

struct kiss_fft_cpx {
	kiss_fft_scalar r;
	kiss_fft_scalar i;
};

#define MAXFACTORS 32
struct kiss_fft_state {
	int nfft;
	int inverse;
	int factors[2 * MAXFACTORS];
	kiss_fft_cpx twiddles[1];
};

typedef struct kiss_fft_state* kiss_fft_cfg;

kiss_fft_cfg kiss_fft_alloc(int nfft, int inverse_fft, void *mem, size_t *lenmem);

void kiss_fft(kiss_fft_cfg cfg, const kiss_fft_cpx *fin, kiss_fft_cpx *fout);

void kiss_fft_stride(kiss_fft_cfg cfg, const kiss_fft_cpx *fin, kiss_fft_cpx *fout, int fin_stride);

#define kiss_fft_free linear_free
//void kiss_fft_cleanup(void);

typedef struct kiss_fftr_state *kiss_fftr_cfg;


kiss_fftr_cfg kiss_fftr_alloc(int nfft, int inverse_fft, void * mem, size_t * lenmem);

void kiss_fftr(kiss_fftr_cfg cfg, const kiss_fft_scalar *timedata, kiss_fft_cpx *freqdata);

void kiss_fftr2(kiss_fftr_cfg st, const kiss_fft_scalar *timedata, kiss_fft_scalar *freqdata);

void kiss_fftri(kiss_fftr_cfg cfg, const kiss_fft_cpx *freqdata, kiss_fft_scalar *timedata);

void kiss_fftri2(kiss_fftr_cfg st, const kiss_fft_scalar *freqdata, kiss_fft_scalar *timedata);
#define kiss_fftr_free linear_free

struct kiss_config {
	kiss_fftr_cfg forward;
	kiss_fftr_cfg backward;
	int N;
};

/** Compute tables for an FFT */
void *linear_fft_init(int size);

/** Destroy tables for an FFT */
void linear_fft_destroy(void *table);

/** Forward (real to half-complex) transform */
void linear_fft(void *table, float *in, float *out);

/** Backward (half-complex to real) transform */
void linear_ifft(void *table, float *in, float *out);

#endif

struct AecCore {
  explicit AecCore(int instance_index);
  ~AecCore();

  std::unique_ptr<ApmDataDumper> data_dumper;
  const OouraFft ooura_fft;

  CoherenceState coherence_state;

  int farBufWritePos, farBufReadPos;

  int knownDelay;
  int inSamples, outSamples;
  int delayEstCtr;

  // Nearend buffer used for changing from FRAME_LEN to PART_LEN sample block
  // sizes. The buffer stores all the incoming bands and for each band a maximum
  // of PART_LEN - (FRAME_LEN - PART_LEN) values need to be buffered in order to
  // change the block size from FRAME_LEN to PART_LEN.
  float nearend_buffer[NUM_HIGH_BANDS_MAX + 1]
                      [PART_LEN - (FRAME_LEN - PART_LEN)];
  size_t nearend_buffer_size;
  float output_buffer[NUM_HIGH_BANDS_MAX + 1][2 * PART_LEN];
  size_t output_buffer_size;

  float eBuf[PART_LEN2];  // error

  float previous_nearend_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN];

  float xPow[PART_LEN1];
  float dPow[PART_LEN1];
  float dMinPow[PART_LEN1];
  float dInitMinPow[PART_LEN1];
  float* noisePow;

  float xfBuf[2][kExtendedNumPartitions * PART_LEN1];  // farend fft buffer
  float wfBuf[2][kExtendedNumPartitions * PART_LEN1];  // filter fft
  // Farend windowed fft buffer.
  complex_t xfwBuf[kExtendedNumPartitions * PART_LEN1];

  float hNs[PART_LEN1];
  float hNlFbMin, hNlFbLocalMin;
  float hNlXdAvgMin;
  int hNlNewMin, hNlMinCtr;
  float overDrive;
  float overdrive_scaling;
  int nlp_mode;
  float outBuf[PART_LEN];
  int delayIdx;

  short stNearState, echoState;
  short divergeState;

  int xfBufBlockPos;

  BlockBuffer farend_block_buffer_;

  int system_delay;  // Current system delay buffered in AEC.

  int mult;  // sampling frequency multiple
  int sampFreq = 16000;
  size_t num_bands;
  uint32_t seed;

  float filter_step_size;  // stepsize
  float error_threshold;   // error threshold

  int noiseEstCtr;

  PowerLevel farlevel;
  PowerLevel nearlevel;
  PowerLevel linoutlevel;
  PowerLevel nlpoutlevel;

  int metricsMode;
  int stateCounter;
  Stats erl;
  Stats erle;
  Stats aNlp;
  Stats rerl;
  DivergentFilterFraction divergent_filter_fraction;

  // Quantities to control H band scaling for SWB input
  int freq_avg_ic;       // initial bin for averaging nlp gain
  int flag_Hband_cn;     // for comfort noise
  float cn_scale_Hband;  // scale for comfort noise in H band

  int delay_metrics_delivered;
  int delay_histogram[kHistorySizeBlocks];
  int num_delay_values;
  int delay_median;
  int delay_std;
  float fraction_poor_delays;
  int delay_logging_enabled;
  void* delay_estimator_farend;
  void* delay_estimator;
  // Variables associated with delay correction through signal based delay
  // estimation feedback.
  int previous_delay;
  int delay_correction_count;
  int shift_offset;
  float delay_quality_threshold;
  int frame_count;

  // 0 = delay agnostic mode (signal based delay correction) disabled.
  // Otherwise enabled.
  int delay_agnostic_enabled;
  // 1 = extended filter mode enabled, 0 = disabled.
  int extended_filter_enabled;
  // 1 = refined filter adaptation aec mode enabled, 0 = disabled.
  bool refined_adaptive_filter_enabled;

  // Runtime selection of number of filter partitions.
  int num_partitions;

  // Flag that extreme filter divergence has been detected by the Echo
  // Suppressor.
  int extreme_filter_divergence;
#ifdef AEC_ROCORD
  FILE *f_aec_near_in;
  FILE *f_aec_far_in;
  FILE *f_aec_near_out;
  FILE *f_aec_linear_out;
  FILE *f_far_in;
  //FILE *f_far_in2;
#endif
#ifdef AUMDF_FILTER
  short nlpFlag;
  float near_buf[NUM_HIGH_BANDS_MAX + 1][PART_LEN * COUNTER];
  LinearEchoState *st;
#endif
};


#ifdef AUMDF_FILTER
void AEC_Process_Core(AecCore* aecpc,
	float farend_extended_block_lowest_band[PART_LEN2],
	float nearend_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN],
	float output_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN]);

void ProcessNearendBlockMdf(
	AecCore* aec,
	int num_partitions,
	int* x_fft_buf_block_pos,
	float farend_extended_block_lowest_band[PART_LEN2],
	float nearend_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN],
	float output_block[NUM_HIGH_BANDS_MAX + 1][PART_LEN]);
#endif

AecCore* WebRtcAec_CreateAec(int instance_count);  // Returns NULL on error.
void WebRtcAec_FreeAec(AecCore* aec);
int WebRtcAec_InitAec(AecCore* aec, int sampFreq);
void WebRtcAec_InitAec_SSE2(void);
#if defined(MIPS_FPU_LE)
void WebRtcAec_InitAec_mips(void);
#endif
#if defined(WEBRTC_HAS_NEON)
void WebRtcAec_InitAec_neon(void);
#endif

void WebRtcAec_BufferFarendBlock(AecCore* aec, const float* farend);
void WebRtcAec_ProcessFrames(AecCore* aec,
                             const float* const* nearend,
                             size_t num_bands,
                             size_t num_samples,
                             int knownDelay,
                             float* const* out);

// A helper function to call adjust the farend buffer size.
// Returns the number of elements the size was decreased with, and adjusts
// |system_delay| by the corresponding amount in ms.
int WebRtcAec_AdjustFarendBufferSizeAndSystemDelay(AecCore* aec,
                                                   int size_decrease);

// Calculates the median, standard deviation and amount of poor values among the
// delay estimates aggregated up to the first call to the function. After that
// first call the metrics are aggregated and updated every second. With poor
// values we mean values that most likely will cause the AEC to perform poorly.
// TODO(bjornv): Consider changing tests and tools to handle constant
// constant aggregation window throughout the session instead.
int WebRtcAec_GetDelayMetricsCore(AecCore* self,
                                  int* median,
                                  int* std,
                                  float* fraction_poor_delays);

// Returns the echo state (1: echo, 0: no echo).
int WebRtcAec_echo_state(AecCore* self);

// Gets statistics of the echo metrics ERL, ERLE, A_NLP.
void WebRtcAec_GetEchoStats(AecCore* self,
                            Stats* erl,
                            Stats* erle,
                            Stats* a_nlp,
                            float* divergent_filter_fraction);

// Sets local configuration modes.
void WebRtcAec_SetConfigCore(AecCore* self,
                             int nlp_mode,
                             int metrics_mode,
                             int delay_logging);

// Non-zero enables, zero disables.
void WebRtcAec_enable_delay_agnostic(AecCore* self, int enable);

// Returns non-zero if delay agnostic (i.e., signal based delay estimation) is
// enabled and zero if disabled.
int WebRtcAec_delay_agnostic_enabled(AecCore* self);

// Turns on/off the refined adaptive filter feature.
void WebRtcAec_enable_refined_adaptive_filter(AecCore* self, bool enable);

// Returns whether the refined adaptive filter is enabled.
bool WebRtcAec_refined_adaptive_filter(const AecCore* self);

// Enables or disables extended filter mode. Non-zero enables, zero disables.
void WebRtcAec_enable_extended_filter(AecCore* self, int enable);

// Returns non-zero if extended filter mode is enabled and zero if disabled.
int WebRtcAec_extended_filter_enabled(AecCore* self);

// Returns the current |system_delay|, i.e., the buffered difference between
// far-end and near-end.
int WebRtcAec_system_delay(AecCore* self);

// Sets the |system_delay| to |value|.  Note that if the value is changed
// improperly, there can be a performance regression.  So it should be used with
// care.
void WebRtcAec_SetSystemDelay(AecCore* self, int delay);

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC_AEC_CORE_H_
