#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMaxPooling.c"
#else

#include "MKLDNN.h"

static void THNN_(SpatialConvolutionMM_MKLDNN_MaxPooling_init)(
          THLongTensor *primitives,
          int N,
          int inC,
          int inH,
          int inW,
          int kH,
          int kW,
          int dH,
          int dW,
          int outC,
          int outH,
          int outW)
{
#if LOG_ENABLE
	fprintf(stderr,"	SpatialConvolutionMM_MKLDNN_MaxPooling_init start, N=%d,inC=%d,inH=%d,inW=%d,kH=%d,kW=%d,outC=%d,outH=%d,outW=%d\n",N,inC,inH,inW,kH,kW,outC,outH,outW );
#endif
	dnnError_t err;

	int inputOffset[dimension - 2 ] = { 0, 0 };
	size_t inputSize[dimension] = 	{inW,inH,inC,N};
	size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
	size_t outputSize[dimension] = 	{outW,outH,outC,N};
	size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };

	size_t kernelSize[2] = { kH, kW };
	size_t kernelStride[2] = { dH, dW };

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
	CHECK_ERR( dnnPoolingCreateForward_F32(&pool1, attributes, dnnAlgorithmPoolingMax,lt_user_input, kernelSize, kernelStride, inputOffset, dnnBorderZeros), err );
	CHECK_ERR( dnnPoolingCreateBackward_F32(&pool_bwd,attributes,dnnAlgorithmPoolingMax,lt_user_input, kernelSize, kernelStride, inputOffset,dnnBorderZeros), err );
#else
	CHECK_ERR( dnnPoolingCreateForward_F32(&pool1, dnnAlgorithmPoolingMax,lt_user_input, kernelSize, kernelStride, inputOffset, dnnBorderZeros), err );
	CHECK_ERR( dnnPoolingCreateBackward_F32(&pool_bwd, dnnAlgorithmPoolingMax,lt_user_input, kernelSize, kernelStride, inputOffset,dnnBorderZeros), err );
#endif
	dnnLayout_t lt_pool1_output = NULL,lt_pool1_input = NULL,
		lt_pool1_workspace = NULL;
	
	real * workspacePtr = NULL;
	
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool1_output, pool1, dnnResourceDst), err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool1_workspace, pool1, dnnResourceWorkspace), err );
	CHECK_ERR( dnnAllocateBuffer_F32((void**)&workspacePtr, lt_pool1_workspace) , err );

#if 0
	real * newOutPtr = NULL;
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool1_input, pool1, dnnResourceSrc), err );
	/*check input layout*/
	int check = dnnLayoutCompare_F32(lt_user_input, lt_pool1_input);
	fprintf(stderr,"	Pooling, dnnLayoutCompare(lt_user_input, lt_pool1_input) = %d \n",check );
	/*check output layout*/
	check = dnnLayoutCompare_F32(lt_user_output, lt_pool1_output);
	fprintf(stderr,"	Pooling, dnnLayoutCompare(lt_user_output, lt_pool1_output) = %d \n",check );

	dnnLayout_t lt_pool_diff = NULL;
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool_diff, pool_bwd,dnnResourceDiffSrc), err );
	/*check diff src layout*/
	int check = dnnLayoutCompare_F32(lt_user_input, lt_pool_diff);
	fprintf(stderr,"	Pooling, dnnLayoutCompare(lt_user_input, lt_pool_diff) = %d \n",check );


	//dnnPrimitive_t cv_pool1_to_user_output = NULL;
	
	//CHECK_ERR( THNN_(init_conversion)(&cv_pool1_to_user_output, &newOutPtr, lt_user_output, lt_pool1_output), err );

	if (!dnnLayoutCompare_F32(lt_user_output, lt_pool1_output)) {
		CHECK_ERR( dnnConversionCreate_F32(&cv_pool1_to_user_output, lt_pool1_output, lt_user_output), err );
		CHECK_ERR( dnnAllocateBuffer_F32((void**)(&newOutPtr), lt_pool1_output), err );
	}

#endif


	//save the dnnPrimitive to THTensor(long int array)
	primitives->storage->data[POOLING_FORWARD] = (long long)pool1;
	primitives->storage->data[POOLING_BACKWARD] = (long long)pool_bwd;
	primitives->storage->data[POOLING_BUF_WORKSPACE] = (long long)workspacePtr;

#if LOG_ENABLE
	fprintf(stderr,"	SpatialConvolutionMM_MKLDNN_MaxPooling_init end.\n" );
#endif
}

void THNN_(SpatialMaxPooling_MKLDNN_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode,
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
  real *indices_data;


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
    /* indices will contain the locations for each output point */
    THTensor_(resize4d)(indices, nbatch, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

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
		THNN_(SpatialConvolutionMM_MKLDNN_MaxPooling_init)(primitives,N,inC,inH,inW,kH,kW,dH,dW,outC,outH,outW);
	}


	dnnPrimitive_t pool1 = (dnnPrimitive_t) (primitives->storage->data[POOLING_FORWARD]);
	real * workspacePtr = (real *) (primitives->storage->data[POOLING_BUF_WORKSPACE]);

	real * resPool1[dnnResourceNumber] = {0};
	resPool1[dnnResourceSrc] = input_data;
	resPool1[dnnResourceDst] = output_data;
	resPool1[dnnResourceWorkspace] = workspacePtr;

	CHECK_ERR( dnnExecute_F32(pool1, (void*)resPool1), err );
#if LOG_ENABLE
	gettimeofday(&end,NULL);
	double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	Pooling MKLDNN time = %.2f ms\n",duration );
#endif
  /* cleanup */
  THTensor_(free)(input);
}

void THNN_(SpatialMaxPooling_MKLDNN_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode,
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
  real *indices_data;

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
  indices_data = THTensor_(data)(indices);

  /* backprop */
  if (input->nDimension == 3)
  {
    THNN_(SpatialMaxPooling_updateGradInput_frame)(gradInput_data, gradOutput_data,
                                                 indices_data,
                                                 nslices,
                                                 iwidth, iheight,
                                                 owidth, oheight,
                                                 dW, dH);
  }
  else
  {
/**************************************MKLDNN interface*****************************************/
	struct timeval start,end;
	gettimeofday(&start,NULL);
	dnnError_t err;

	dnnPrimitive_t pool_bwd = (dnnPrimitive_t) (primitives->storage->data[POOLING_BACKWARD]);
	real * workspacePtr = (real *) (primitives->storage->data[POOLING_BUF_WORKSPACE]);

	real * resPool1[dnnResourceNumber] = {0};
	resPool1[dnnResourceDiffSrc] = THTensor_(data)(gradInput);
	//resPool1[dnnResourceFilter] = output_data;
	resPool1[dnnResourceWorkspace] = workspacePtr;
	resPool1[dnnResourceDiffDst] = THTensor_(data)(gradOutput);;

	CHECK_ERR( dnnExecute_F32(pool_bwd, (void*)resPool1), err );
#if LOG_ENABLE
	gettimeofday(&end,NULL);
	double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	Pooling MKLDNN time = %.2f ms\n",duration );
#endif
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
}


static void THNN_(SpatialMaxPooling_updateOutput_frame)(
          real *input_p,
          real *output_p,
          real *ind_p,
          long nslices,
          long iwidth,
          long iheight,
          long owidth,
          long oheight,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    long i, j;
    real *ip = input_p   + k*iwidth*iheight;
    for(i = 0; i < oheight; i++)
    {
      for(j = 0; j < owidth; j++)
      {
        long hstart = i * dH - padH;
        long wstart = j * dW - padW;
        long hend = fminf(hstart + kH, iheight);
        long wend = fminf(wstart + kW, iwidth);
        hstart = fmaxf(hstart, 0);
        wstart = fmaxf(wstart, 0);

        /* local pointers */
        real *op = output_p  + k*owidth*oheight + i*owidth + j;
        real *indp = ind_p   + k*owidth*oheight + i*owidth + j;

        /* compute local max: */
        long maxindex = -1;
        real maxval = -THInf;
        long tcntr = 0;
        long x,y;
        for(y = hstart; y < hend; y++)
        {
          for(x = wstart; x < wend; x++)
          {
            tcntr = y*iwidth + x;
            real val = *(ip + tcntr);
            if (val > maxval)
            {
              maxval = val;
              maxindex = tcntr;
            }
          }
        }

        /* set output to local max */
        *op = maxval;

        /* store location of max */
        *indp = maxindex + 1;
      }
    }
  }
}

void THNN_(SpatialMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode)
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
  real *indices_data;


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

  /* resize output */
  if (input->nDimension == 3)
  {
    THTensor_(resize3d)(output, nslices, oheight, owidth);
    /* indices will contain the locations for each output point */
    THTensor_(resize3d)(indices,  nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

    THNN_(SpatialMaxPooling_updateOutput_frame)(input_data, output_data,
                                              indices_data,
                                              nslices,
                                              iwidth, iheight,
                                              owidth, oheight,
                                              kW, kH, dW, dH,
                                              padW, padH);
  }
  else
  {
    long p;

    THTensor_(resize4d)(output, nbatch, nslices, oheight, owidth);
    /* indices will contain the locations for each output point */
    THTensor_(resize4d)(indices, nbatch, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_(SpatialMaxPooling_updateOutput_frame)(input_data+p*nslices*iwidth*iheight, output_data+p*nslices*owidth*oheight,
                                                indices_data+p*nslices*owidth*oheight,
                                                nslices,
                                                iwidth, iheight,
                                                owidth, oheight,
                                                kW, kH, dW, dH,
                                                padW, padH);
    }
  }

  /* cleanup */
  THTensor_(free)(input);
}

static void THNN_(SpatialMaxPooling_updateGradInput_frame)(
          real *gradInput_p,
          real *gradOutput_p,
          real *ind_p,
          long nslices,
          long iwidth,
          long iheight,
          long owidth,
          long oheight,
          int dW,
          int dH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    real *gradInput_p_k = gradInput_p + k*iwidth*iheight;
    real *gradOutput_p_k = gradOutput_p + k*owidth*oheight;
    real *ind_p_k = ind_p + k*owidth*oheight;

    /* calculate max points */
    long i, j;
    for(i = 0; i < oheight; i++)
    {
      for(j = 0; j < owidth; j++)
      {
        /* retrieve position of max */
        long maxp = ind_p_k[i*owidth + j] - 1;
        /* update gradient */
        gradInput_p_k[maxp] += gradOutput_p_k[i*owidth + j];
      }
    }
  }
}

void THNN_(SpatialMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode)
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
  real *indices_data;

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
  indices_data = THTensor_(data)(indices);

  /* backprop */
  if (input->nDimension == 3)
  {
    THNN_(SpatialMaxPooling_updateGradInput_frame)(gradInput_data, gradOutput_data,
                                                 indices_data,
                                                 nslices,
                                                 iwidth, iheight,
                                                 owidth, oheight,
                                                 dW, dH);
  }
  else
  {
    long p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_(SpatialMaxPooling_updateGradInput_frame)(gradInput_data+p*nslices*iwidth*iheight, gradOutput_data+p*nslices*owidth*oheight,
                                                   indices_data+p*nslices*owidth*oheight,
                                                   nslices,
                                                   iwidth, iheight,
                                                   owidth, oheight,
                                                   dW, dH);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
}

#endif
