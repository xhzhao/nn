#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Threshold.c"
#else

#include "MKLDNN.h"
static void THNN_(SpatialConvolutionMM_MKLDNN_Relu_init)(
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
	dnnLayout_t lt_relu_input = NULL,lt_relu_diff_out=NULL;

#if NEW_INTERFACE
	/*for new interface*/
	dnnPrimitiveAttributes_t attributes = NULL;
	CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );
#endif
	size_t inputSize[dimension] = 	{inW,inH,inC,N};
	size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
	if(primitives->storage->data[RELU_LAYOUT_INPUT] == 0)
	{
		CHECK_ERR( dnnLayoutCreate_F32(&lt_relu_input, dimension, inputSize, inputStrides) , err );
		primitives->storage->data[RELU_LAYOUT_OUTPUT] = lt_relu_input;
		fprintf(stderr ,"MKLDNN RELU fail to get input layout \n");
	}
	else{
		lt_relu_input = primitives->storage->data[RELU_LAYOUT_INPUT];
		primitives->storage->data[RELU_LAYOUT_OUTPUT] = primitives->storage->data[RELU_LAYOUT_INPUT];
		fprintf(stderr ,"MKLDNN RELU get valid input layout \n");
	}



	size_t outputSize[dimension] = 	{outW,outH,outC,N};
	size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };
	CHECK_ERR( dnnLayoutCreate_F32(&lt_relu_diff_out, dimension, outputSize, outputStrides) , err );



#if NEW_INTERFACE
	CHECK_ERR( dnnReLUCreateForward_F32(&relu_forward, attributes, lt_relu_input, threshold), err );
	CHECK_ERR( dnnReLUCreateBackward_F32(&relu_backward, attributes, lt_relu_diff_out, lt_relu_input, threshold), err );
#else
	CHECK_ERR( dnnReLUCreateForward_F32(&relu1, lt_relu_input, threshold), err );
	CHECK_ERR( dnnReLUCreateBackward_F32(&relu1,lt_relu_diff_out, lt_relu_input, threshold), err );
#endif
	primitives->storage->data[RELU_FORWARD] = (long long)relu_forward;
	primitives->storage->data[RELU_BACKWARD] = (long long)relu_backward;

}

void THNN_(Threshold_MKLDNN_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real threshold,
          real val,
          bool inplace,
          THLongTensor *primitives,
          int initOk)
{
	struct timeval start,end;
	gettimeofday(&start,NULL);
	dnnError_t err;
	dnnPrimitive_t relu1 = NULL;
	dnnLayout_t lt_relu_input = NULL;

#if LOG_ENABLE
	fprintf(stderr, "MKLDNN Relu forward start\n");
	//fprintf(stderr, "MKLDNN Relu forward start:inplace=%d, N=%d,inC=%d,inH=%d,inW=%d, inPtr=%d, outPtr=%d \n",inplace,N,inC,inH,inW,inPtr,outPtr);
#endif

	THTensor_(resizeAs)(output, input);
	int N = input->size[0];
	int inC = input->size[1];
	int inH = input->size[2];
	int inW = input->size[3];
	real * inPtr = THTensor_(data)(input);
	real * outPtr = THTensor_(data)(output);


	int outC = output->size[1];
	int outH = output->size[2];
	int outW = output->size[3];
	
	if(initOk == 0)
	{
		THNN_(SpatialConvolutionMM_MKLDNN_Relu_init)(primitives,N,inC,inH,inW,outC,outH,outW,threshold);
	}
	
	relu1 = (dnnPrimitive_t) (primitives->storage->data[RELU_FORWARD]);



/*
	size_t inputSize[dimension] = 	{inW,inH,inC,N};
	size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
	CHECK_ERR( dnnLayoutCreate_F32(&lt_relu_input, dimension, inputSize, inputStrides) , err );
#if NEW_INTERFACE
	CHECK_ERR( dnnReLUCreateForward_F32(&relu1, attributes, lt_relu_input, threshold), err );
#else
	CHECK_ERR( dnnReLUCreateForward_F32(&relu1, lt_relu_input, threshold), err );
#endif
*/

	/*check output layout*/
/*	dnnLayout_t 
	int check = dnnLayoutCompare_F32(lt_pr, lt_us)
*/

	real *resRelu1[dnnResourceNumber];
	resRelu1[dnnResourceSrc] = inPtr;
	resRelu1[dnnResourceDst] = outPtr;

	CHECK_ERR( dnnExecute_F32(relu1, (void**)resRelu1), err );

#if LOG_ENABLE
	fprintf(stderr, "MKLDNN Relu forward end \n");
#endif
}

void THNN_(Threshold_MKLDNN_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real threshold,
          bool inplace,
          THLongTensor *primitives)
{
/*
  if (inplace)
  {
    TH_TENSOR_APPLY2(real, gradOutput, real, input,
      if ((*input_data) <= threshold)
        *gradOutput_data = 0;
    );
    THTensor_(set)(gradInput, gradOutput);
  }
  else
*/
  {
	struct timeval start,end;
	gettimeofday(&start,NULL);
	dnnError_t err;
	dnnPrimitive_t relu1 = NULL;
	dnnLayout_t lt_relu_input = NULL,lt_relu_diff_out=NULL;



	THTensor_(resizeAs)(gradInput, input);
	relu1 = (dnnPrimitive_t) (primitives->storage->data[RELU_BACKWARD]);

	real *resRelu1[dnnResourceNumber];
	resRelu1[dnnResourceSrc] 	= THTensor_(data)(input);
	resRelu1[dnnResourceDiffSrc] 	= THTensor_(data)(gradInput);
	resRelu1[dnnResourceDiffDst] 	= THTensor_(data)(gradOutput);

	CHECK_ERR( dnnExecute_F32(relu1, (void**)resRelu1), err );


#if LOG_ENABLE
	gettimeofday(&end,NULL);
	double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	Relu backward time = %.2f ms\n",duration );
#endif
   }
}



void THNN_(Threshold_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real threshold,
          real val,
          bool inplace)
{
  if (inplace)
  {
    TH_TENSOR_APPLY(real, input,
      if (*input_data <= threshold)
        *input_data = val;
    );
    THTensor_(set)(output, input);
  }
  else
  {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(real, output, real, input,
      *output_data = (*input_data > threshold) ? *input_data : val;
    );
  }
}

void THNN_(Threshold_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real threshold,
          bool inplace)
{
  if (inplace)
  {
    TH_TENSOR_APPLY2(real, gradOutput, real, input,
      if ((*input_data) <= threshold)
        *gradOutput_data = 0;
    );
    THTensor_(set)(gradInput, gradOutput);
  }
  else
  {
    THTensor_(resizeAs)(gradInput, input);
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
      if ((*input_data) > threshold)
        *gradInput_data = *gradOutput_data;
      else
        *gradInput_data = 0;
    );
  }
}

#endif
