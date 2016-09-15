
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ConcatMKLDNN.c"
#else

#include "MKLDNN.h"
static void THNN_(Concat_MKLDNN_init_forward)(
          THLongTensor *primitives,
          int N,
          int inC,
          int inH,
          int inW,
          int outC,
          int outH,
          int outW,
	  real threshold
	  )
{
	dnnError_t err;
	dnnPrimitive_t relu_forward = NULL, relu_backward = NULL;
	dnnLayout_t lt_relu_input = NULL,lt_relu_diff_out=NULL, lt_relu_forward_output;
	real * buffer_forward_output = NULL;

}

static void THNN_(Concat_MKLDNN_init_backward)(
          THLongTensor *primitives,
          int N,
          int inC,
          int inH,
          int inW,
          int outC,
          int outH,
          int outW,
	  real threshold
	  )
{
}

void THNN_(Concat_MKLDNN_setupLongTensor)(
          THNNState *state,
          THLongTensor * array,
          THTensor *input,
          int  index)
{
	fprintf(stderr, "Concat_MKLDNN_setupLongTensor start. \n");
	array->storage->data[index] = (long long )array;
}


/**
input: the long tensor , the tensor size = moduleNum, the data is the THTensor ptr which point to the real data
*/
void THNN_(Concat_MKLDNN_updateOutput)(
          THNNState *state,
          THLongTensor *input,
          THTensor *output,
          int  moduleNum,
          THLongTensor *primitives,
          int initOk)
{
	fprintf(stderr, "Concat_MKLDNN_updateOutput start. \n");
	long long inputPtr = input->storage->data[0];
	THTensor * input0 = (THTensor *)inputPtr;
}

void THNN_(Concat_MKLDNN_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real threshold,
          bool inplace,
          THLongTensor *primitives,
          int initOk)
{

}

#endif
