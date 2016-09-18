
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ConcatMKLDNN.c"
#else

#include "MKLDNN.h"
static void THNN_(Concat_MKLDNN_init_forward)(
          THLongTensor *inputarray,
          THTensor *output,
          int  moduleNum,
          THLongTensor *primitives
	  )
{
#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_init_forward start. \n");
#endif
	dnnError_t err;
	dnnPrimitive_t m_concat_forward = NULL;
	THTensor * input = NULL;
	long long inputPtr = 0;
	dnnLayout_t *layouts = malloc(moduleNum * sizeof(dnnLayout_t));
	for(int i=0; i < moduleNum; i++)
	{
		inputPtr = inputarray->storage->data[i];
		input = (THTensor *)inputPtr;


		if(input->mkldnnLayout == 0)
		{
#if CONVERSION_LOG
			fprintf(stderr, "Concat MKLDNN get input layout fail, i = %d \n", i);
#endif
			//create NCHW layout here
			int N = input->size[0];
			int inC = input->size[1];
			int inH = input->size[2];
			int inW = input->size[3];
			dnnLayout_t lt_user_input = NULL;
			size_t inputSize[dimension] = 	{inW,inH,inC,N};
			size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
			CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides) , err );
			layouts[i] = lt_user_input;
		}
		else
		{
#if CONVERSION_LOG
			int N = input->size[0];
			int inC = input->size[1];
			int inH = input->size[2];
			int inW = input->size[3];
			fprintf(stderr, "Concat MKLDNN get input layout OK, N = %d, inC = %d, inH = %d, inW = %d \n",N,inC,inH,inW);
#endif
			layouts[i] = (dnnLayout_t)input->mkldnnLayout;
		}
	}
	CHECK_ERR(dnnConcatCreate_F32(&m_concat_forward, NULL, moduleNum, layouts), err);

	primitives->storage->data[CONCAT_FORWARD] = (long long)m_concat_forward;


#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_init_forward end. \n");
#endif
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
#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_setupLongTensor start, array = 0x%x, input = 0x%x, index = %d\n", array, input, index);
#endif
	array->storage->data[index-1] = (long long )input;

}


/**
input: the long tensor , the tensor size = moduleNum, the data is the THTensor ptr which point to the real data
*/
void THNN_(Concat_MKLDNN_updateOutput)(
          THNNState *state,
          THLongTensor *inputarray,
          THTensor *output,
          int  moduleNum,
          THLongTensor *primitives,
          int initOk)
{
#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_updateOutput start. inputarray = 0x%x, output = 0x%d, moduleNum = %d,  primitives = 0x%x, initOk = %d \n", inputarray, output, moduleNum, primitives, initOk);
#endif
	dnnError_t err;

	if(initOk == 0)
	{
		THNN_(Concat_MKLDNN_init_forward)(inputarray, output, moduleNum, primitives);
	}

	dnnPrimitive_t m_concat_forward = NULL;
	THTensor * input = NULL;
	long long inputPtr = 0;
	dnnLayout_t *layouts = NULL;
	void *concat_res[dnnResourceNumber];
	for(int i=0; i < moduleNum; i++)
	{
		inputPtr = inputarray->storage->data[i];
		input = (THTensor *)inputPtr;
		concat_res[dnnResourceMultipleSrc + i] = THTensor_(data)(input);
		
	}
	concat_res[dnnResourceDst] = THTensor_(data)(output);
	m_concat_forward = (dnnPrimitive_t) (primitives->storage->data[CONCAT_FORWARD]);

	CHECK_ERR( dnnExecute_F32(m_concat_forward, (void*)concat_res), err );
	
	
#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_updateOutput end. \n");
#endif

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
