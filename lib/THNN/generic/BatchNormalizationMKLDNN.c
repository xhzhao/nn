#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/BatchNormalizationMKLDNN.c"
#else

#include "MKLDNN.h"

static void THNN_(BatchNormalization_MKLDNN_init_forward)(
          THLongTensor *primitives,
          int N,
          int inC,
          int inH,
          int inW,
	  double eps)
{
	dnnError_t err;
	dnnPrimitive_t bn_forward = NULL;
	dnnPrimitive_t bn_backward = NULL;
	dnnPrimitive_t bn_bwd_scaleshift = NULL;

	size_t inputSize[dimension] = 	{inW,inH,inC,N};
	size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };

	dnnLayout_t lt_user_input = NULL;

	if(primitives->storage->data[BN_LAYOUT_INPUT] == 0)
	{
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides) , err );
		primitives->storage->data[BN_LAYOUT_OUTPUT] = lt_user_input;
		fprintf(stderr ,"MKLDNN BN get input layout FAIL......\n");
	}
	else{
		lt_user_input = primitives->storage->data[BN_LAYOUT_INPUT];
		primitives->storage->data[BN_LAYOUT_OUTPUT] = primitives->storage->data[BN_LAYOUT_INPUT];
		fprintf(stderr ,"MKLDNN BN get input layout OK\n");
	}

	CHECK_ERR( dnnBatchNormalizationCreateForward_F32(&bn_forward,NULL,lt_user_input,eps), err );
	CHECK_ERR( dnnBatchNormalizationCreateBackwardData_F32(&bn_backward,NULL,lt_user_input,eps), err );
	CHECK_ERR( dnnBatchNormalizationCreateBackwardScaleShift_F32(&bn_bwd_scaleshift,NULL,lt_user_input,eps), err );
	

	dnnLayout_t lt_bn_forward_workspace,lt_bn_forward_scaleshift,lt_bn_forward_output,lt_bn_backward_input;
	real * buffer_forward_workspace = NULL; real * buffer_forward_scaleshift = NULL;
	dnnLayoutCreateFromPrimitive_F32(&lt_bn_forward_workspace, bn_forward, dnnResourceWorkspace);
	dnnLayoutCreateFromPrimitive_F32(&lt_bn_forward_output, bn_forward, dnnResourceDst);
	dnnLayoutCreateFromPrimitive_F32(&lt_bn_forward_scaleshift, bn_forward, dnnResourceScaleShift);
	dnnLayoutCreateFromPrimitive_F32(&lt_bn_backward_input, bn_backward, dnnResourceDiffSrc);
		

	CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forward_workspace), lt_bn_forward_workspace), err );
	CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forward_scaleshift), lt_bn_forward_scaleshift), err );
	
	//save the dnnPrimitive to THTensor(long int array)
	primitives->storage->data[BN_LAYOUT_FORWARD_OUTPUT] = (long long)lt_bn_forward_output;
	primitives->storage->data[BN_LAYOUT_BACKWARD_INPUT] = (long long)lt_bn_backward_input;

	primitives->storage->data[BN_FORWARD] = (long long)bn_forward;
	primitives->storage->data[BN_BACKWARD] = (long long)bn_backward;
	primitives->storage->data[BN_SCALESHIFT] = (long long)bn_bwd_scaleshift;
	primitives->storage->data[BUFFER_BN_FORWARD_WORKSPACE] = (long long)buffer_forward_workspace;
	primitives->storage->data[BUFFER_BN_FORWARD_SCALESHIFT] = (long long)buffer_forward_scaleshift;
	primitives->storage->data[BUFFER_BN_BACKWARD_WORKSPACE] = (long long)buffer_forward_workspace;
}

static void THNN_(BatchNormalization_MKLDNN_init_backward)(
          THLongTensor *primitives,
          int N,
          int outC,
          int outH,
          int outW,
	  double eps)
{
	dnnError_t err;

	dnnPrimitive_t bn_backward = primitives->storage->data[BN_BACKWARD];
	size_t outputSize[dimension] = 	{outW,outH,outC,N};
	size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };

	dnnLayout_t lt_user_output,lt_bn_backward_output=NULL;

	if(primitives->storage->data[BN_LAYOUT_OUTPUT] == 0)
	{
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides) , err );
		fprintf(stderr ,"MKLDNN BN get input layout FAIL......\n");
	}
	else{
		lt_user_output = primitives->storage->data[BN_LAYOUT_OUTPUT];
		primitives->storage->data[BN_LAYOUT_OUTPUT] = primitives->storage->data[BN_LAYOUT_OUTPUT];
		fprintf(stderr ,"MKLDNN BN get input layout OK\n");
	}

	dnnLayoutCreateFromPrimitive_F32(&lt_bn_backward_output, bn_backward, dnnResourceDiffDst);
	dnnPrimitive_t cv_backward_output = NULL;real * buffer_backward_output = NULL;
	//backward conversion init
	CHECK_ERR( THNN_(init_conversion)(&cv_backward_output, &buffer_backward_output, lt_bn_backward_output, lt_user_output), err );

	//save the dnnPrimitive to THTensor(long int array)
	primitives->storage->data[CV_BN_BACKWARD_OUTPUT] = (long long)cv_backward_output;
	primitives->storage->data[BUFFER_BN_BACKWARD_OUTPUT] = (long long)buffer_backward_output;
}



void THNN_(BatchNormalization_MKLDNN_updateOutput)(
  THNNState *state, THTensor *input, THTensor *output,
  THTensor *weight, THTensor *bias,
  THTensor *running_mean, THTensor *running_var,
  THTensor *save_mean, THTensor *save_std,
  bool train, double momentum, double eps,
  THLongTensor *primitives,int initOk)
{
  long nInput = THTensor_(size)(input, 1);
  long f,n = THTensor_(nElement)(input) / nInput;

	dnnError_t err;
	int N = input->size[0];
	int inC = input->size[1];
	int inH = input->size[2];
	int inW = input->size[3];


	if(initOk == 0)
	{
		primitives->storage->data[BN_LAYOUT_INPUT] = input->mkldnnLayout;
		THNN_(BatchNormalization_MKLDNN_init_forward)(primitives,N,inC,inH,inW,eps);
	}
	dnnPrimitive_t bn_forward = primitives->storage->data[BN_FORWARD];
	real * buffer_forward_workspace = primitives->storage->data[BUFFER_BN_FORWARD_WORKSPACE];
	real * buffer_forward_scaleshift = primitives->storage->data[BUFFER_BN_FORWARD_SCALESHIFT];

	//fprintf(stderr, "BN MKLDNN, nInput = %d \n", nInput);
	for(int i =0; i < inC; i++)
	{
		buffer_forward_scaleshift[i] = weight ? THTensor_(get1d)(weight, i) : 1;
		buffer_forward_scaleshift[i+inC] = bias ? THTensor_(get1d)(bias, i) : 0;
	}

  	void* BatchNorm_res[dnnResourceNumber];
	BatchNorm_res[dnnResourceSrc] = THTensor_(data)(input);
	BatchNorm_res[dnnResourceDst] = THTensor_(data)(output);
	BatchNorm_res[dnnResourceWorkspace] = buffer_forward_workspace;
	BatchNorm_res[dnnResourceScaleShift] = buffer_forward_scaleshift;

	CHECK_ERR( dnnExecute_F32(bn_forward, (void*)BatchNorm_res), err );
	output->mkldnnLayout = primitives->storage->data[BN_LAYOUT_FORWARD_OUTPUT];
	
}

void THNN_(BatchNormalization_MKLDNN_backward)(
  THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradInput,
  THTensor *gradWeight, THTensor *gradBias, THTensor *weight,
  THTensor *running_mean, THTensor *running_var,
  THTensor *save_mean, THTensor *save_std,
  bool train, double scale, double eps,
  THLongTensor *primitives,
          int initOk)
{
  long nInput = THTensor_(size)(input, 1);
  long f,n = THTensor_(nElement)(input) / nInput;

	dnnError_t err;
	int inC = input->size[1];
	dnnPrimitive_t bn_backward = primitives->storage->data[BN_BACKWARD];
	dnnPrimitive_t bn_bwd_scaleshift = primitives->storage->data[BN_SCALESHIFT];
	real * buffer_forward_workspace = primitives->storage->data[BUFFER_BN_FORWARD_WORKSPACE];
	real * buffer_forward_scaleshift = primitives->storage->data[BUFFER_BN_FORWARD_SCALESHIFT];


	if(gradInput == 0)
	{
		void* BatchNormScaleshift_res[dnnResourceNumber];
		BatchNormScaleshift_res[dnnResourceSrc] = THTensor_(data)(input);
		BatchNormScaleshift_res[dnnResourceDiffDst] = THTensor_(data)(gradOutput);
		BatchNormScaleshift_res[dnnResourceDiffSrc] = THTensor_(data)(gradInput);
		BatchNormScaleshift_res[dnnResourceWorkspace] = buffer_forward_workspace;
		BatchNormScaleshift_res[dnnResourceScaleShift] = buffer_forward_scaleshift;
		fprintf(stderr, "bn_bwd_scaleshift exec start \n");
		fprintf(stderr, "BatchNormalization_MKLDNN_backward filter, input=0x%x,gradOutput=0x%x,gradInput=0x%x,workspace=0x%x,scaleshift=0x%x \n", THTensor_(data)(input),THTensor_(data)(gradOutput),THTensor_(data)(gradInput),buffer_forward_workspace,buffer_forward_scaleshift);
		CHECK_ERR( dnnExecute_F32(bn_bwd_scaleshift, (void*)BatchNormScaleshift_res), err );
		fprintf(stderr, "bn_bwd_scaleshift exec done \n");
		for(int i=0; i < inC; i++)
		{
			THTensor_(set1d)(gradWeight, i, buffer_forward_scaleshift[i]);
			THTensor_(set1d)(gradBias, i, buffer_forward_scaleshift[i+inC]);
		}
	}else
	{

		if(initOk == 0)
		{
			int N = gradOutput->size[0];
			int outC = gradOutput->size[1];
			int outH = gradOutput->size[2];
			int outW = gradOutput->size[3];

			primitives->storage->data[BN_LAYOUT_OUTPUT] = gradOutput->mkldnnLayout;
			THNN_(BatchNormalization_MKLDNN_init_backward)(primitives,N,outC,outH,outW,eps);
		}
		dnnPrimitive_t cv_backward_output = (dnnPrimitive_t) (primitives->storage->data[CV_BN_BACKWARD_OUTPUT]);

		real * buffer_backward_output = (dnnPrimitive_t) (primitives->storage->data[BUFFER_BN_BACKWARD_OUTPUT]);

		void* BatchNorm_res[dnnResourceNumber];
		BatchNorm_res[dnnResourceSrc] = THTensor_(data)(input);
		BatchNorm_res[dnnResourceDiffDst] = THTensor_(data)(gradOutput);
		BatchNorm_res[dnnResourceDiffSrc] = THTensor_(data)(gradInput);
		BatchNorm_res[dnnResourceWorkspace] = buffer_forward_workspace;
		BatchNorm_res[dnnResourceScaleShift] = buffer_forward_scaleshift;

		if(cv_backward_output)
		{
			fprintf(stderr, "	Relu backward output conversion... \n");
			BatchNorm_res[dnnResourceDiffDst] = buffer_backward_output;
			CHECK_ERR( dnnConversionExecute_F32(cv_backward_output, THTensor_(data)(gradOutput), BatchNorm_res[dnnResourceDiffDst]), err );
		}

		CHECK_ERR( dnnExecute_F32(bn_backward, (void*)BatchNorm_res), err );
		//fprintf(stderr, "bn_backward exec done");
		gradInput->mkldnnLayout = primitives->storage->data[BN_LAYOUT_BACKWARD_INPUT];

	}


}

#endif
