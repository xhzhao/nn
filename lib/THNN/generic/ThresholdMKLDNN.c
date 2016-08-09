#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ThresholdMKLDNN.c"
#else

#include "MKLDNN.h"
static void THNN_(SpatialConvolutionMM_MKLDNN_Relu_init_forward)(
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
		fprintf(stderr ,"MKLDNN RELU get input layout FAIL......\n");
	}
	else{
		lt_relu_input = (dnnLayout_t)primitives->storage->data[RELU_LAYOUT_INPUT];
		fprintf(stderr ,"MKLDNN RELU get input layout OK\n");
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

	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_relu_forward_output, relu_forward, dnnResourceDst), err );

	primitives->storage->data[RELU_FORWARD] = (long long)relu_forward;
	primitives->storage->data[RELU_BACKWARD] = (long long)relu_backward;
	primitives->storage->data[RELU_LAYOUT_FORWARD_OUTPUT] = (long long)lt_relu_forward_output;


}

static void THNN_(SpatialConvolutionMM_MKLDNN_Relu_init_backward)(
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
	dnnPrimitive_t relu_backward = (dnnPrimitive_t) (primitives->storage->data[RELU_BACKWARD]);
	dnnLayout_t lt_relu_diff_out=NULL,lt_relu_diff_src=NULL,lt_user_output=NULL;
	dnnPrimitive_t cv_backward_output = NULL;real * buffer_backward_output = NULL;
#if NEW_INTERFACE
	/*for new interface*/
	dnnPrimitiveAttributes_t attributes = NULL;
	CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );
#endif
	size_t outputSize[dimension] = 	{outW,outH,outC,N};
	size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };
	if(primitives->storage->data[RELU_LAYOUT_OUTPUT] == 0)
	{
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides) , err );
		fprintf(stderr ,"MKLDNN RELU get output layout FAIL......\n");
	}
	else{
		lt_user_output = (dnnLayout_t)primitives->storage->data[RELU_LAYOUT_OUTPUT];
		fprintf(stderr ,"MKLDNN RELU get output layout OK\n");
	}

	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_relu_diff_out, relu_backward, dnnResourceDiffDst), err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_relu_diff_src, relu_backward, dnnResourceDiffSrc), err );

	//backward conversion init
	CHECK_ERR( THNN_(init_conversion)(&cv_backward_output, &buffer_backward_output, lt_relu_diff_out, lt_user_output), err );

	primitives->storage->data[CV_RELU_BACKWARD_OUTPUT] = (long long)cv_backward_output;
	primitives->storage->data[BUFFER_RELU_BACKWARD_OUTPUT] = (long long)buffer_backward_output;
	primitives->storage->data[RELU_LAYOUT_BACKWARD_INPUT] = (long long)lt_relu_diff_src;
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
		primitives->storage->data[RELU_LAYOUT_INPUT] = (long long)input->mkldnnLayout;
		THNN_(SpatialConvolutionMM_MKLDNN_Relu_init_forward)(primitives,N,inC,inH,inW,outC,outH,outW,threshold);
	}
	
	relu1 = (dnnPrimitive_t) (primitives->storage->data[RELU_FORWARD]);

	real *resRelu1[dnnResourceNumber];
	resRelu1[dnnResourceSrc] = inPtr;
	resRelu1[dnnResourceDst] = outPtr;

	CHECK_ERR( dnnExecute_F32(relu1, (void**)resRelu1), err );

	//output->storage->data = resRelu1[dnnResourceDst];
	//output->storageOffset = 0;
	output->mkldnnLayout = primitives->storage->data[RELU_LAYOUT_FORWARD_OUTPUT];
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
          THLongTensor *primitives,
          int initOk)
{
	struct timeval start,end;
	gettimeofday(&start,NULL);
	dnnError_t err;
	dnnPrimitive_t relu1 = NULL;
	dnnLayout_t lt_relu_input = NULL,lt_relu_diff_out=NULL;
	dnnPrimitive_t cv_backward_output = NULL;real * buffer_backward_output = NULL;

	int N = input->size[0];
	int inC = input->size[1];
	int inH = input->size[2];
	int inW = input->size[3];
	int outC = gradOutput->size[1];
	int outH = gradOutput->size[2];
	int outW = gradOutput->size[3];
	
	if(initOk == 0)
	{
		primitives->storage->data[RELU_LAYOUT_OUTPUT] = (long long)gradOutput->mkldnnLayout;
		THNN_(SpatialConvolutionMM_MKLDNN_Relu_init_backward)(primitives,N,inC,inH,inW,outC,outH,outW,threshold);
	}

	THTensor_(resizeAs)(gradInput, input);
	relu1 = (dnnPrimitive_t) (primitives->storage->data[RELU_BACKWARD]);

	real *resRelu1[dnnResourceNumber];
	resRelu1[dnnResourceSrc] 	= THTensor_(data)(input);
	resRelu1[dnnResourceDiffSrc] 	= THTensor_(data)(gradInput);
	resRelu1[dnnResourceDiffDst] 	= THTensor_(data)(gradOutput);

	cv_backward_output = (dnnPrimitive_t)primitives->storage->data[CV_RELU_BACKWARD_OUTPUT];
	buffer_backward_output = (real *)primitives->storage->data[BUFFER_RELU_BACKWARD_OUTPUT];
	if(cv_backward_output)
	{
		fprintf(stderr, "	RELU backward output conversion \n");
		resRelu1[dnnResourceDiffDst] = buffer_backward_output;
		CHECK_ERR( dnnConversionExecute_F32(cv_backward_output, THTensor_(data)(gradOutput), resRelu1[dnnResourceDiffDst]), err );
	}
	CHECK_ERR( dnnExecute_F32(relu1, (void**)resRelu1), err );
	if(cv_backward_output)
	{
		gradInput->storage->data = resRelu1[dnnResourceDiffSrc];
		gradInput->storageOffset = 0;
	}
	gradInput->mkldnnLayout = (long long)primitives->storage->data[RELU_LAYOUT_BACKWARD_INPUT];
	
#if LOG_ENABLE
	gettimeofday(&end,NULL);
	double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	Relu backward time = %.2f ms\n",duration );
#endif
}

#endif
