#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/BatchNormalizationMKLDNN.c"
#else

#include "MKLDNN.h"

static void THNN_(BatchNormalization_MKLDNN_init)(
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
		fprintf(stderr ,"MKLDNN BN fail to get input layout \n");
	}
	else{
		lt_user_input = primitives->storage->data[BN_LAYOUT_INPUT];
		primitives->storage->data[BN_LAYOUT_OUTPUT] = primitives->storage->data[BN_LAYOUT_INPUT];
		fprintf(stderr ,"MKLDNN BN get valid input layout \n");
	}

	CHECK_ERR( dnnBatchNormalizationCreateForward_F32(&bn_forward,NULL,lt_user_input,eps), err );
	CHECK_ERR( dnnBatchNormalizationCreateBackwardData_F32(&bn_backward,NULL,lt_user_input,eps), err );
	CHECK_ERR( dnnBatchNormalizationCreateBackwardScaleShift_F32(&bn_bwd_scaleshift,NULL,lt_user_input,eps), err );
	

	dnnLayout_t lt_bn_forward_workspace,lt_bn_forward_scaleshift;
	real * buffer_forward_workspace = NULL; real * buffer_forward_scaleshift = NULL;
	dnnLayoutCreateFromPrimitive_F32(&lt_bn_forward_workspace, bn_forward, dnnResourceWorkspace);
	dnnLayoutCreateFromPrimitive_F32(&lt_bn_forward_scaleshift, bn_forward, dnnResourceScaleShift);
	
	
	CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forward_workspace), lt_bn_forward_workspace), err );
	CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forward_scaleshift), lt_bn_forward_scaleshift), err );
	
	//save the dnnPrimitive to THTensor(long int array)
	primitives->storage->data[BN_FORWARD] = (long long)bn_forward;
	primitives->storage->data[BN_BACKWARD] = (long long)bn_backward;
	primitives->storage->data[BN_SCALESHIFT] = (long long)bn_bwd_scaleshift;
	primitives->storage->data[BUFFER_BN_FORWARD_WORKSPACE] = (long long)buffer_forward_workspace;
	primitives->storage->data[BUFFER_BN_FORWARD_SCALESHIFT] = (long long)buffer_forward_scaleshift;
	primitives->storage->data[BUFFER_BN_BACKWARD_WORKSPACE] = (long long)buffer_forward_workspace;

	
	
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
		THNN_(BatchNormalization_MKLDNN_init)(primitives,N,inC,inH,inW,eps);
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


	//fprintf(stderr, "BatchNormalization_MKLDNN_updateOutput, input=0x%x,output=0x%x,workspace=0x%x,scaleshift=0x%x \n", THTensor_(data)(input),THTensor_(data)(output),buffer_forward_workspace,buffer_forward_scaleshift);
  void* BatchNorm_res[dnnResourceNumber];
  BatchNorm_res[dnnResourceSrc] = THTensor_(data)(input);
  BatchNorm_res[dnnResourceDst] = THTensor_(data)(output);
  BatchNorm_res[dnnResourceWorkspace] = buffer_forward_workspace;
  BatchNorm_res[dnnResourceScaleShift] = buffer_forward_scaleshift;

	CHECK_ERR( dnnExecute_F32(bn_forward, (void*)BatchNorm_res), err );
}

void THNN_(BatchNormalization_MKLDNN_backward)(
  THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradInput,
  THTensor *gradWeight, THTensor *gradBias, THTensor *weight,
  THTensor *running_mean, THTensor *running_var,
  THTensor *save_mean, THTensor *save_std,
  bool train, double scale, double eps,
  THLongTensor *primitives)
{
  long nInput = THTensor_(size)(input, 1);
  long f,n = THTensor_(nElement)(input) / nInput;

	dnnError_t err;
	int inC = input->size[1];
	dnnPrimitive_t bn_backward = primitives->storage->data[BN_BACKWARD];
	dnnPrimitive_t bn_bwd_scaleshift = primitives->storage->data[BN_SCALESHIFT];
	real * buffer_forward_workspace = primitives->storage->data[BUFFER_BN_FORWARD_WORKSPACE];
	real * buffer_forward_scaleshift = primitives->storage->data[BUFFER_BN_FORWARD_SCALESHIFT];


	if(gradWeight && gradBias)
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
		void* BatchNorm_res[dnnResourceNumber];
		BatchNorm_res[dnnResourceSrc] = THTensor_(data)(input);
		BatchNorm_res[dnnResourceDiffDst] = THTensor_(data)(gradOutput);
		BatchNorm_res[dnnResourceDiffSrc] = THTensor_(data)(gradInput);
		BatchNorm_res[dnnResourceWorkspace] = buffer_forward_workspace;
		BatchNorm_res[dnnResourceScaleShift] = buffer_forward_scaleshift;

		//fprintf(stderr, "BatchNormalization_MKLDNN_backward data, input=0x%x,gradOutput=0x%x,gradInput=0x%x,workspace=0x%x,scaleshift=0x%x \n", THTensor_(data)(input),THTensor_(data)(gradOutput),THTensor_(data)(gradInput),buffer_forward_workspace,buffer_forward_scaleshift);
		CHECK_ERR( dnnExecute_F32(bn_backward, (void*)BatchNorm_res), err );
		//fprintf(stderr, "bn_backward exec done");
	}


}

#endif
