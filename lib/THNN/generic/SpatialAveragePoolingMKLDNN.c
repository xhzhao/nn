#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialAveragePoolingMKLDNN.c"
#else

#include "MKLDNN.h"

static void THNN_(SpatialAveragePooling_MKLDNN_init)(
          THLongTensor *primitives,
          int N,
          int inC,
          int inH,
          int inW,
          int kH,
          int kW,
          int dH,
          int dW,
	  int padH,
	  int padW,
          int outC,
          int outH,
          int outW)
{
#if 1 //LOG_ENABLE
	fprintf(stderr,"	SpatialAveragePooling_MKLDNN_init start, N=%d,inC=%d,inH=%d,inW=%d,kH=%d,kW=%d,dH=%d,dW=%d,outC=%d,outH=%d,outW=%d\n",N,inC,inH,inW,kH,kW,dH,dW,outC,outH,outW );
#endif
	dnnError_t err;

	int inputOffset[dimension - 2 ] = { 0, 0 };
	size_t inputSize[dimension] = 	{inW,inH,inC,N};
	size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
	size_t outputSize[dimension] = 	{outW,outH,outC,N};
	size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };

	size_t kernelSize[2] = { kH, kW };
	size_t kernelStride[2] = { dH, dW };
	int pad[dimension-2] = 	{-padW,-padH};


	real * resPool1[dnnResourceNumber] = {0};
	dnnLayout_t lt_user_input = NULL,lt_user_output=NULL;
	CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides) , err );
	CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides) , err );

#if NEW_INTERFACE
	/*for new interface*/
	dnnPrimitiveAttributes_t attributes = NULL;
	CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );
#endif


	dnnPrimitive_t pool1 = NULL;
	dnnPrimitive_t pool_bwd = NULL;
#if NEW_INTERFACE
	CHECK_ERR( dnnPoolingCreateForward_F32(&pool1, attributes, dnnAlgorithmPoolingAvg,lt_user_input, kernelSize, kernelStride, pad, dnnBorderZeros), err );
	CHECK_ERR( dnnPoolingCreateBackward_F32(&pool_bwd,attributes,dnnAlgorithmPoolingAvg,lt_user_input, kernelSize, kernelStride, pad,dnnBorderZeros), err );
#else
	CHECK_ERR( dnnPoolingCreateForward_F32(&pool1, dnnAlgorithmPoolingAvg,lt_user_input, kernelSize, kernelStride, inputOffset, dnnBorderZeros), err );
	CHECK_ERR( dnnPoolingCreateBackward_F32(&pool_bwd, dnnAlgorithmPoolingAvg,lt_user_input, kernelSize, kernelStride, inputOffset,dnnBorderZeros), err );
#endif
	dnnLayout_t lt_pool_forward_output = NULL,lt_pool_forward_input = NULL,lt_pool_forward_workspace = NULL;
	dnnLayout_t lt_pool_backward_output = NULL,lt_pool_backward_input = NULL,lt_pool_backward_workspace = NULL;

	dnnPrimitive_t cv_forward_input = NULL,cv_forward_output = NULL;
	dnnPrimitive_t cv_backward_input = NULL,cv_backward_output = NULL;

	real * buffer_forward_input = NULL;	real * buffer_forward_output = NULL;	real * buffer_forward_workspace = NULL;
	real * buffer_backward_input = NULL;	real * buffer_backward_output = NULL;	real * buffer_backward_workspace = NULL;


	
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool_forward_input, pool1, dnnResourceSrc), err );	
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool_forward_output, pool1, dnnResourceDst), err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool_forward_workspace, pool1, dnnResourceWorkspace), err );
	CHECK_ERR( dnnAllocateBuffer_F32((void**)&buffer_forward_workspace, lt_pool_forward_workspace) , err );

	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool_backward_input, pool_bwd, dnnResourceDiffSrc), err );	
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool_backward_output, pool_bwd, dnnResourceDiffDst), err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool_backward_workspace, pool_bwd, dnnResourceWorkspace), err );
	CHECK_ERR( dnnAllocateBuffer_F32((void**)&buffer_backward_workspace, lt_pool_backward_workspace) , err );

	//forward conversion init
	//CHECK_ERR( THNN_(init_conversion)(&cv_forward_input, &buffer_forward_input, lt_pool_forward_input, lt_user_input), err );
	if(!dnnLayoutCompare_F32(lt_user_output, lt_pool_forward_output))
	{
		fprintf(stderr, "cv_forward_output = 0x%x, lt_pool_forward_output = 0x%x, lt_user_output=0x%x \n",cv_forward_output,lt_pool_forward_output,lt_user_output);
		CHECK_ERR( dnnConversionCreate_F32(&cv_forward_output, lt_pool_forward_output, lt_user_output), err );
		CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forward_output), lt_pool_forward_output), err );
	}

	//backward conversion init
	CHECK_ERR( THNN_(init_conversion)(&cv_backward_output, &buffer_backward_output, lt_pool_backward_output, lt_user_output), err );
	if(!dnnLayoutCompare_F32(lt_user_input, lt_pool_backward_input))
	{
		CHECK_ERR( dnnConversionCreate_F32(&cv_backward_input, lt_pool_backward_input, lt_user_input), err );
		CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_backward_input), lt_pool_backward_input), err );
	}


	//save the dnnPrimitive to THTensor(long int array)
	primitives->storage->data[POOLING_FORWARD] = (long long)pool1;
	primitives->storage->data[POOLING_BACKWARD] = (long long)pool_bwd;
	primitives->storage->data[CV_POOLING_FORWARD_INPUT] = (long long)cv_forward_input;
	primitives->storage->data[CV_POOLING_FORWARD_OUTPUT] = (long long)cv_forward_output;
	primitives->storage->data[CV_POOLING_BACKWARD_INPUT] = (long long)cv_backward_input;
	primitives->storage->data[CV_POOLING_BACKWARD_OUTPUT] = (long long)cv_backward_output;


	primitives->storage->data[BUFFER_POOLING_FORWARD_INPUT] = (long long)buffer_forward_input;
	primitives->storage->data[BUFFER_POOLING_FORWARD_OUTPUT] = (long long)buffer_forward_output;
	primitives->storage->data[BUFFER_POOLING_FORWARD_WORKSPACE] = (long long)buffer_forward_workspace;

	primitives->storage->data[BUFFER_POOLING_BACKWARD_INPUT] = (long long)buffer_backward_input;
	primitives->storage->data[BUFFER_POOLING_BACKWARD_OUTPUT] = (long long)buffer_backward_output;
	primitives->storage->data[BUFFER_POOLING_BACKWARD_WORKSPACE] = (long long)buffer_backward_workspace;


#if LOG_ENABLE
	fprintf(stderr,"	SpatialConvolutionMM_MKLDNN_AveragePooling_init end.\n" );
#endif
}




void THNN_(SpatialAveragePooling_MKLDNN_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode,
          bool count_include_pad,
          THLongTensor *primitives,
          int initOk)

{
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  long nslices;
  long iheight;
  long iwidth;
  long oheight;
  long owidth;
  real *input_data;
  real *output_data;


  THArgCheck(input->nDimension == 3 || input->nDimension == 4 , 2, "3D or 4D (batch mode) tensor expected");

  if (input->nDimension == 4) 
  {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }
  THArgCheck(input->size[dimw] >= kW - padW && input->size[dimh] >= kH - padH, 2, "input image smaller than kernel size");

  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  if (ceil_mode)
  {
    oheight = (long)(ceil((float)(iheight - kH + 2*padH) / dH)) + 1;
    owidth  = (long)(ceil((float)(iwidth  - kW + 2*padW) / dW)) + 1;
  }
  else
  {
    oheight = (long)(floor((float)(iheight - kH + 2*padH) / dH)) + 1;
    owidth  = (long)(floor((float)(iwidth  - kW + 2*padW) / dW)) + 1;
  }

  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    if ((oheight - 1)*dH >= iheight + padH)
      --oheight;
    if ((owidth  - 1)*dW >= iwidth  + padW)
      --owidth;
  }

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

    THTensor_(resize4d)(output, nbatch, nslices, oheight, owidth);
    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);

/**************************************MKLDNN interface*****************************************/
	struct timeval start,end;
	gettimeofday(&start,NULL);
	dnnError_t err;

	int N = input->size[0];
	int inC = input->size[1];
	int inH = input->size[2];
	int inW = input->size[3];

	int outC = output->size[1];
	int outH = output->size[2];
	int outW = output->size[3];

	if(initOk == 0)
	{
		THNN_(SpatialAveragePooling_MKLDNN_init)(primitives,N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW);
	}

	dnnPrimitive_t cv_forward_input = NULL,cv_forward_output = NULL;
	real * buffer_forward_input = NULL;	real * buffer_forward_output = NULL;	real * buffer_forward_workspace = NULL;


	dnnPrimitive_t pool1 	= (dnnPrimitive_t) (primitives->storage->data[POOLING_FORWARD]);
	cv_forward_input 	= (dnnPrimitive_t) (primitives->storage->data[CV_POOLING_FORWARD_INPUT]);
	cv_forward_output 	= (dnnPrimitive_t) (primitives->storage->data[CV_POOLING_FORWARD_OUTPUT]);
	buffer_forward_input	= (dnnPrimitive_t) (primitives->storage->data[BUFFER_POOLING_FORWARD_INPUT]);
	buffer_forward_output	= (dnnPrimitive_t) (primitives->storage->data[BUFFER_POOLING_FORWARD_OUTPUT]);
	buffer_forward_workspace= (dnnPrimitive_t) (primitives->storage->data[BUFFER_POOLING_FORWARD_WORKSPACE]);


	real * resPool1[dnnResourceNumber] = {0};
	resPool1[dnnResourceSrc] = input_data;
	resPool1[dnnResourceDst] = output_data;
	resPool1[dnnResourceWorkspace] = buffer_forward_workspace;

/*	if(cv_forward_input)
	{
			resPool1[dnnResourceSrc] = buffer_forward_input;
			CHECK_ERR( dnnConversionExecute_F32(cv_forward_input, input_data, resPool1[dnnResourceSrc]), err );
	}*/
	if(cv_forward_output){
		fprintf(stderr, "	AveragePooling , need to convert the output \n");
		resPool1[dnnResourceDst] = buffer_forward_output;
	}

	CHECK_ERR( dnnExecute_F32(pool1, (void*)resPool1), err );
	if(cv_forward_output){
		CHECK_ERR( dnnConversionExecute_F32(cv_forward_output, buffer_forward_output, output_data), err );
	} 

#if LOG_ENABLE
	gettimeofday(&end,NULL);
	double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	Pooling MKLDNN time = %.2f ms\n",duration );
#endif
  /* cleanup */
  THTensor_(free)(input);
}




void THNN_(SpatialAveragePooling_MKLDNN_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode,
          bool count_include_pad,
          THLongTensor *primitives)
{
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  int nslices;
  int iheight;
  int iwidth;
  int oheight;
  int owidth;
  real *gradInput_data;
  real *gradOutput_data;

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  oheight = gradOutput->size[dimh];
  owidth = gradOutput->size[dimw];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);

  /* backprop */
  if (input->nDimension == 3)
  {
	fprintf(stderr, "Error: MKLDNN SpatialAveragePooling don't support dimension=3 \n");
  }
  else
  {
/**************************************MKLDNN interface*****************************************/

	struct timeval start,end;
	gettimeofday(&start,NULL);
	dnnError_t err;

	dnnPrimitive_t pool_bwd = (dnnPrimitive_t) (primitives->storage->data[POOLING_BACKWARD]);
	dnnPrimitive_t cv_backward_input = NULL,cv_backward_output = NULL;
	real * buffer_backward_input = NULL;	real * buffer_backward_output = NULL;	real * buffer_backward_workspace = NULL;

	cv_backward_input 	= (dnnPrimitive_t) (primitives->storage->data[CV_POOLING_BACKWARD_INPUT]);
	cv_backward_output 	= (dnnPrimitive_t) (primitives->storage->data[CV_POOLING_BACKWARD_OUTPUT]);
	buffer_backward_input	= (dnnPrimitive_t) (primitives->storage->data[BUFFER_POOLING_BACKWARD_INPUT]);
	buffer_backward_output	= (dnnPrimitive_t) (primitives->storage->data[BUFFER_POOLING_BACKWARD_OUTPUT]);
	buffer_backward_workspace= (dnnPrimitive_t) (primitives->storage->data[BUFFER_POOLING_BACKWARD_WORKSPACE]);


	real * resPool1[dnnResourceNumber] = {0};
	resPool1[dnnResourceDiffSrc] = gradInput_data;
	resPool1[dnnResourceDiffDst] = gradOutput_data;
	resPool1[dnnResourceWorkspace] = buffer_backward_workspace;

	if(cv_backward_output)
	{
		fprintf(stderr, "	Averagepooling backward output conversion...");
		resPool1[dnnResourceDiffDst] = buffer_backward_output;
		CHECK_ERR( dnnConversionExecute_F32(cv_backward_output, gradOutput_data, resPool1[dnnResourceDiffDst]), err );
	}
	if(cv_backward_input){
		resPool1[dnnResourceDiffSrc] = buffer_backward_input;
	}

	CHECK_ERR( dnnExecute_F32(pool_bwd, (void*)resPool1), err );
	if(cv_backward_input){
		fprintf(stderr, "	Averagepooling backward input conversion...");
		CHECK_ERR( dnnConversionExecute_F32(cv_backward_output, buffer_backward_input, gradInput_data), err );
	}


#if LOG_ENABLE
	gettimeofday(&end,NULL);
	double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	Pooling MKLDNN time = %.2f ms\n",duration );
#endif
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
}


#endif
