#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionMM.c"
#else


#include "MKLDNN.h"

static dnnError_t  THNN_(init_conversion)(dnnPrimitive_t *cv, real **ptr_out,
                                 dnnLayout_t lt_pr, dnnLayout_t lt_us)
{
	dnnError_t err;
	*ptr_out = NULL;
	if(sizeof(real) == sizeof(float))
	{
		
		if (!dnnLayoutCompare_F32(lt_pr, lt_us)) {
			//fprintf(stderr, "compare fail, need to do conversion.\n");
			CHECK_ERR( dnnConversionCreate_F32(cv, lt_us, lt_pr), err );
			CHECK_ERR( dnnAllocateBuffer_F32((void**)ptr_out, lt_pr), err );
		}
		else
		{
			//fprintf(stderr, "compare ok.\n");
		}
		return E_SUCCESS;
	}
	else if(sizeof(real) == sizeof(double))
	{
		if (!dnnLayoutCompare_F64(lt_pr, lt_us)) {
		CHECK_ERR( dnnConversionCreate_F64(cv, lt_us, lt_pr), err );
		CHECK_ERR( dnnAllocateBuffer_F64((void**)ptr_out, lt_pr), err );
		}
		return E_SUCCESS;
	}

}


void THNN_(SpatialConvolutionMM_compare)(
          THNNState * state,
          THTensor * mkldnn,
          THTensor * old,
          long long len,
	  int compareSource
	)
{
	real * ptr1 = THTensor_(data)(mkldnn);
	real * ptr2 = THTensor_(data)(old);
	long long i = 0;
	for(i=0; i< len; i++)
	{
		if( (ptr1[i]-ptr2[i] > 0.01 ) || (ptr1[i]-ptr2[i] < -0.01 ))
		{
			break;
		}
	}

	if(i == len)
	{
//#if LOG_ENABLE
		fprintf(stderr, "	compareSource = %d, mkldnn is same as old, good. len =%ld, mkldnn[0]=%.3f,mkldnn[1]=%.3f,mkldnn[2]=%.3f, mkldnn[15]=%.3f, old[15]=%.3f\n",
  			compareSource,len,ptr1[0],ptr1[1],ptr1[2], ptr1[15], ptr2[15]);
//#endif	
	}
	else
	{
		fprintf(stderr, "	compareSource = %d, compare fail: i =%d, len=%d, mkldnn[i] = %.4f, old[i]=%.4f \n", compareSource, i, len, ptr1[i], ptr2[i]);
	}

}



static void THNN_(SpatialConvolutionMM_MKLDNN_init)(
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
	dnnError_t err;
#if LOG_ENABLE
	fprintf(stderr, "SpatialConvolutionMM_MKLDNN_init: start.");
	fprintf(stderr, "N=%d,inC=%d,inH=%d,inW=%d,kH=%d,kW=%d,dH=%d,dW=%d,padH=%d,padW=%d,outC=%d,outH=%d,outW=%d\n", N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW);
#endif
	dnnPrimitive_t m_conv_forward = NULL;
	dnnPrimitive_t m_conv_bwd_data = NULL;
	dnnPrimitive_t m_conv_bwd_filter = NULL;

	size_t inputSize[dimension] = 	{inW,inH,inC,N};
	size_t filterSize[dimension] = 	{kW,kH,inC,outC};
	size_t outputSize[dimension] = 	{outW,outH,outC,N};
	size_t stride[dimension-2] = 	{dW,dH};
	int pad[dimension-2] = 		{-padW,-padH};
	int compareResult[4] = {2};


	size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };
	size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
	size_t filterStrides[dimension] = { 1, kW, kH * kW, inC * kH * kW };

	size_t biasSize[1] = { outputSize[2] };
	size_t biasStrides[1] = { 1 };
	//size_t biasStrides[1] = { outputStrides[2] };

	dnnLayout_t lt_user_input, lt_user_filt, lt_user_bias, lt_user_output;
	// Convolution describes what layout it expects
	dnnLayout_t lt_conv1_input, lt_conv1_filt, lt_conv1_bias, lt_conv1_output;

	dnnLayout_t lt_conv_diff_src = NULL;	//for backward data
	dnnLayout_t lt_conv_diff_filter = NULL; //for backward filter

	dnnPrimitive_t cv_user_to_conv1_input = NULL,
		cv_user_to_conv1_filt = NULL,
		cv_user_to_conv1_bias = NULL,
		cv_user_to_conv1_output = NULL,
		cv_conv_to_user_output = NULL,
		cv_conv_to_user_input = NULL,
		cv_conv_to_user_filt = NULL;
	real * newInPtr = NULL;
	real * newFilterPtr = NULL;
	real * newBiasPtr = NULL;
	real * newOutPtr = NULL;

#if NEW_INTERFACE
/*for new interface*/
	dnnPrimitiveAttributes_t attributes = NULL;
	CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );
#endif
/**/


	if(sizeof(real) == sizeof(float))
	{
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides) , err );
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_filt, dimension, filterSize, filterStrides), err );
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_bias, 1, biasSize, biasStrides) , err );
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides), err );

#if NEW_INTERFACE
		CHECK_ERR(dnnConvolutionCreateForwardBias_F32(&m_conv_forward, attributes, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
		CHECK_ERR(dnnConvolutionCreateBackwardData_F32(&m_conv_bwd_data, attributes, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
		CHECK_ERR(dnnConvolutionCreateBackwardFilter_F32(&m_conv_bwd_filter, attributes, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
#else
		CHECK_ERR(dnnConvolutionCreateForwardBias_F32(&m_conv_forward, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
		CHECK_ERR(dnnConvolutionCreateBackwardData_F32(&m_conv_bwd_data, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
		CHECK_ERR(dnnConvolutionCreateBackwardFilter_F32(&m_conv_bwd_filter, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);

#endif

		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv1_input, m_conv_forward, dnnResourceSrc) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv1_filt, m_conv_forward, dnnResourceFilter), err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv1_bias, m_conv_forward, dnnResourceBias) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv1_output,m_conv_forward, dnnResourceDst) , err );	


		CHECK_ERR( THNN_(init_conversion)(&cv_user_to_conv1_input, &newInPtr, lt_conv1_input, lt_user_input) , err );
		CHECK_ERR( THNN_(init_conversion)(&cv_user_to_conv1_filt, &newFilterPtr, lt_conv1_filt, lt_user_filt), err );
		CHECK_ERR( THNN_(init_conversion)(&cv_user_to_conv1_bias, &newBiasPtr, lt_conv1_bias, lt_user_bias), err );


		if (!dnnLayoutCompare_F32(lt_user_output, lt_conv1_output)) {
			//fprintf(stderr, "compare fail, need to do conversion.\n");
			CHECK_ERR( dnnConversionCreate_F32(&cv_conv_to_user_output, lt_conv1_output, lt_user_output), err );
			CHECK_ERR( dnnConversionCreate_F32(&cv_user_to_conv1_output, lt_user_output, lt_conv1_output), err );
			CHECK_ERR( dnnAllocateBuffer_F32((void**)(&newOutPtr), lt_conv1_output), err );
		}
		
		
   		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv_diff_src, m_conv_bwd_data,dnnResourceDiffSrc), err );
		if (!dnnLayoutCompare_F32(lt_user_input, lt_conv_diff_src)) {
			//fprintf(stderr, "compare fail, need to do conversion.\n");
			CHECK_ERR( dnnConversionCreate_F32(&cv_conv_to_user_input, lt_conv_diff_src, lt_user_input), err );
		}


		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv_diff_filter, m_conv_bwd_filter,dnnResourceDiffFilter), err );
		if (!dnnLayoutCompare_F32(lt_user_filt, lt_conv_diff_filter)) {
			//fprintf(stderr, "compare fail, need to do conversion.\n");
			CHECK_ERR( dnnConversionCreate_F32(&cv_conv_to_user_filt, lt_conv_diff_filter, lt_user_filt), err );
		}

	}
	else if(sizeof(real) == sizeof(double))
	{
#if NEW_INTERFACE
		CHECK_ERR(dnnConvolutionCreateForward_F64(&m_conv_forward,attributes, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
		CHECK_ERR(dnnConvolutionCreateBackwardData_F64(&m_conv_bwd_data,attributes, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
		CHECK_ERR(dnnConvolutionCreateBackwardFilter_F64(&m_conv_bwd_filter,attributes, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
#else
		CHECK_ERR(dnnConvolutionCreateForward_F64(&m_conv_forward, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
		CHECK_ERR(dnnConvolutionCreateBackwardData_F64(&m_conv_bwd_data, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
		CHECK_ERR(dnnConvolutionCreateBackwardFilter_F64(&m_conv_bwd_filter, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
#endif
	}

	//save the dnnPrimitive to THTensor(long int array)
	primitives->storage->data[FORWARD_INDEX] = (long long)m_conv_forward;
	primitives->storage->data[BWD_DATA_INDEX] = (long long)m_conv_bwd_data;
	primitives->storage->data[BWD_FILTER_INDEX] = (long long)m_conv_bwd_filter;

	primitives->storage->data[CONVERT_INPUT_INDEX] = (long long)cv_user_to_conv1_input;
	primitives->storage->data[CONVERT_FILTER_INDEX] = (long long)cv_user_to_conv1_filt;
	primitives->storage->data[CONVERT_BIAS_INDEX] = (long long)cv_user_to_conv1_bias;
	primitives->storage->data[CONVERT_CONV_OUTPUT_INDEX] = (long long)cv_user_to_conv1_output;

	primitives->storage->data[CONVERT_OUTPUT_INDEX] = (long long)cv_conv_to_user_output;
	primitives->storage->data[CONVERT_BWDDATA_INDEX] = (long long)cv_conv_to_user_input;
	primitives->storage->data[CONVERT_BWDFILTER_INDEX] = (long long)cv_conv_to_user_filt;


	primitives->storage->data[BUF_CONVERT_INPUT_INDEX] = (long long)newInPtr;
	primitives->storage->data[BUF_CONVERT_FILTER_INDEX] = (long long)newFilterPtr;
	primitives->storage->data[BUF_CONVERT_BIAS_INDEX] = (long long)newBiasPtr;
	primitives->storage->data[BUF_CONVERT_OUTPUT_INDEX] = (long long)newOutPtr;

/*
	long long temp = (long long)(newInPtr);
	real * tempPtr = (real *)(temp);

	fprintf(stderr, "		cv1=0x%x, cv2=0x%x, cv3=0x%x \n",cv_user_to_conv1_input,cv_user_to_conv1_filt,cv_user_to_conv1_bias);
	fprintf(stderr, "		buf1=0x%x, buf2=0x%x, buf3=0x%x \n",newInPtr,newFilterPtr,newBiasPtr);
	fprintf(stderr, "		sizeof(newInPtr) = %d\n",sizeof(newInPtr));
	if(cv_user_to_conv1_input) 	fprintf(stderr, "	newInPtr[0]=%.2f,newInPtr[1]=%.2f,newInPtr[10]=%.2f. \n",tempPtr[0],tempPtr[1],tempPtr[10]);
*/
	//fprintf(stderr, "		cv1=0x%x, cv2=0x%x, cv3=0x%x, cv4=0x%x \n",cv_user_to_conv1_input,cv_user_to_conv1_filt,cv_user_to_conv1_bias,cv_user_to_conv1_output);
#if LOG_ENABLE
	fprintf(stderr, "SpatialConvolutionMM_MKLDNN_init: end, sizeof(real)=%d\n",sizeof(real));
#endif

}



void THNN_(SpatialConvolutionMM_MKLDNN_forward)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          THLongTensor *primitives,
          int initOk,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
	struct timeval start,end;
	gettimeofday(&start,NULL);
	dnnError_t err;
	dnnPrimitive_t m_conv_forward = NULL;

	int N = input->size[0];
	int inC = input->size[1];
	int inH = input->size[2];
	int inW = input->size[3];

	int outC = weight->size[0];
	int outH = (inH + 2*padH - kH)/dH + 1;
	int outW = (inW + 2*padW - kW)/dW + 1;

	dnnPrimitive_t cv_user_to_conv1_input = NULL,
		cv_user_to_conv1_filt = NULL,
		cv_user_to_conv1_bias = NULL,
		cv_conv_to_user_output = NULL;
	real * newInPtr = NULL;
	real * newFilterPtr = NULL;
	real * newBiasPtr = NULL;
	real * newOutPtr = NULL;

	if(initOk == 0)
	{
		THNN_(SpatialConvolutionMM_MKLDNN_init)(primitives,N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW);
	}
	m_conv_forward = (dnnPrimitive_t) (primitives->storage->data[FORWARD_INDEX]);
	cv_user_to_conv1_input 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_INPUT_INDEX];
	cv_user_to_conv1_filt 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_FILTER_INDEX];
	cv_user_to_conv1_bias 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BIAS_INDEX];
	cv_conv_to_user_output  = (dnnPrimitive_t)primitives->storage->data[CONVERT_OUTPUT_INDEX];
	newInPtr 		= (real *)(primitives->storage->data[BUF_CONVERT_INPUT_INDEX]);
	newFilterPtr 		= (real *)(primitives->storage->data[BUF_CONVERT_FILTER_INDEX]);
	newBiasPtr 		= (real *)(primitives->storage->data[BUF_CONVERT_BIAS_INDEX]);
	newOutPtr 		= (real *)(primitives->storage->data[BUF_CONVERT_OUTPUT_INDEX]);


	THTensor_(resize3d)(finput, N, kW*kH*inC, outH*outW);
	THTensor_(resize4d)(output, N, outC, outH, outW);
	//return;
#if LOG_ENABLE
	fprintf(stderr, "SpatialConvolutionMM_MKLDNN_forward: start, m_conv_forward = 0x%x \n",m_conv_forward);
	fprintf(stderr, "	input->size[0]=%d,input->size[1]=%d,input->size[2]=%d,input->size[3]=%d \n", input->size[0],input->size[1],input->size[2],input->size[3]);	
	fprintf(stderr, "	output->size[0]=%d,output->size[1]=%d,output->size[2]=%d,output->size[3]=%d \n", output->size[0],output->size[1],output->size[2],output->size[3]);
	fprintf(stderr, "	weight->size[0]=%d,weight->size[1]=%d\n", weight->size[0],weight->size[1]);
	fprintf(stderr, "	bias->nDimension=%d,bias->size[0]=%d,bias->storage->data[0]=%.3f\n", bias->nDimension,bias->size[0],bias->storage->data[0]);
#endif

/**/
	long long i = 0;
	real * inPtr = THTensor_(data)(input);
	real * filterPtr = THTensor_(data)(weight);
	real * outPtr = THTensor_(data)(output);
	real * biasPtr = THTensor_(data)(bias);

    //fprintf(stderr, "	input->offset=%d, weight->offset=%d, output->offset=%d, bias->offset=%d\n",input->storageOffset,weight->storageOffset, output->storageOffset,bias->storageOffset);
/*
	inPtr = (real *) calloc(N*inC*inH*inW, sizeof(real));
	filterPtr = (real *) calloc(outC*inC*kH*kW, sizeof(real));
	biasPtr = (real *) calloc(bias->size[0], sizeof(real));
	outPtr = (real *) calloc(N*outC*outH*outW, sizeof(real));
*/
	//fprintf(stderr, "input[0]=%.2f,input[1]=%.2f,input[2]=%.2f,input[3]=%.2f,input[4]=%.2f\n",inPtr[0],inPtr[1],inPtr[2],inPtr[3],inPtr[4]);

/*	for(i=0; i < input->size[0]*input->size[1]*input->size[2]*input->size[3]; i++)
	{
		inPtr[i] = 1;
	}
	fprintf(stderr, "		input data init ok. \n");
	for(i=0; i < output->size[0]*output->size[1]*output->size[2]*output->size[3]; i++)
	{
		outPtr[i] = 0;
	}
	fprintf(stderr, "		output data init ok. \n");

	for(i=0; i < weight->size[0]*weight->size[1]; i++)
	{
		filterPtr[i] = 3;
	}
	fprintf(stderr, "		weight data init ok. \n");

	fprintf(stderr, "	filterPtr = 0x%x, biasPtr = 0x%x\n", filterPtr, biasPtr);
	fprintf(stderr, "	bias->nDimension=%d,bias->size[0]=%d,bias->storage->data[0]=%.3f\n", bias->nDimension,bias->size[0],bias->storage->data[0]);
	biasPtr = (real *) calloc(10*bias->size[0], sizeof(real));
	for(i=0; i < bias->size[0]; i++)
	{
		biasPtr[i] = 0;
	}
	fprintf(stderr, "		bias data init ok. \n");

*/


	real * resConv[dnnResourceNumber]={0};
	if(sizeof(real) == sizeof(float))
	{
		//fprintf(stderr, "		cv1=0x%x, cv2=0x%x, cv3=0x%x, cv4=0x%x\n",cv_user_to_conv1_input,cv_user_to_conv1_filt,cv_user_to_conv1_bias,cv_conv_to_user_output);
		//fprintf(stderr, "		buf1=0x%x, buf2=0x%x, buf3=0x%x, buf4=0x%x \n",newInPtr,newFilterPtr,newBiasPtr,newOutPtr);
		if(cv_user_to_conv1_input){
			resConv[dnnResourceSrc] = newInPtr;
			//fprintf(stderr, "	newInPtr[0]=%.2f,newInPtr[1]=%.2f,newInPtr[10]=%.2f. \n",newInPtr[0],newInPtr[1],newInPtr[10]);
			CHECK_ERR( dnnConversionExecute_F32(cv_user_to_conv1_input, inPtr, resConv[dnnResourceSrc]), err );
			//fprintf(stderr,"	conversion 1 ok.\n" );
		} 
		else{
			resConv[dnnResourceSrc] = inPtr;
		}
		
		if(cv_user_to_conv1_filt){
			resConv[dnnResourceFilter] = newFilterPtr;
			CHECK_ERR( dnnConversionExecute_F32(cv_user_to_conv1_filt, filterPtr, resConv[dnnResourceFilter]), err );
			//fprintf(stderr,"	conversion 2 ok.\n" );
		} 
		else{
			resConv[dnnResourceFilter] = filterPtr;
		}
		
		if(cv_user_to_conv1_bias){
			resConv[dnnResourceBias] = newBiasPtr;
			CHECK_ERR( dnnConversionExecute_F32(cv_user_to_conv1_bias, biasPtr, resConv[dnnResourceBias]), err );
			fprintf(stderr,"	conversion 3 ok.\n" );
		} 
		else{
			resConv[dnnResourceBias] = biasPtr;
		}
		if(cv_conv_to_user_output){
			resConv[dnnResourceDst] = newOutPtr;
		} 
		else{
			resConv[dnnResourceDst] = outPtr;
		}
		//fprintf(stderr, "		call float api:dnnExecute_F32 start . \n");

		CHECK_ERR(dnnExecute_F32(m_conv_forward, (void**)resConv),err);	
		if(cv_conv_to_user_output){
			CHECK_ERR( dnnConversionExecute_F32(cv_conv_to_user_output, newOutPtr, outPtr), err );
			//fprintf(stderr,"	conversion 4 ok.\n" );
		} 
		//fprintf(stderr, "		call float api:dnnExecute_F32 end, out[0]=%.2f \n",outPtr[0]);
	}
	else if(sizeof(real) == sizeof(double))
	{

		CHECK_ERR(dnnExecute_F64(m_conv_forward, (void**)resConv),err);
	}
#if LOG_ENABLE
	gettimeofday(&end,NULL);
	double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	forward MKLDNN time = %.2f ms\n",duration );
#endif
}


void THNN_(SpatialConvolutionMM_MKLDNN_bwdData)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          THLongTensor *primitives,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
	struct timeval start,end;
	gettimeofday(&start,NULL);
	dnnError_t err;
	dnnPrimitive_t m_conv_bwdData =NULL; 
	dnnPrimitive_t cv_user_to_conv1_filt = NULL,
		cv_user_to_conv1_output = NULL,
		cv_conv_to_user_input=NULL;
	real * newInPtr = NULL;
	real * newFilterPtr = NULL;
	real * newBiasPtr = NULL;
	real * newOutPtr = NULL;

	m_conv_bwdData = (dnnPrimitive_t) (primitives->storage->data[BWD_DATA_INDEX]);

	cv_user_to_conv1_filt 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_FILTER_INDEX];
	cv_user_to_conv1_output  = (dnnPrimitive_t)primitives->storage->data[CONVERT_CONV_OUTPUT_INDEX];
	cv_conv_to_user_input	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDDATA_INDEX];
	newInPtr 		= (real *)(primitives->storage->data[BUF_CONVERT_INPUT_INDEX]);
	newFilterPtr 		= (real *)(primitives->storage->data[BUF_CONVERT_FILTER_INDEX]);
	newBiasPtr 		= (real *)(primitives->storage->data[BUF_CONVERT_BIAS_INDEX]);
	newOutPtr 		= (real *)(primitives->storage->data[BUF_CONVERT_OUTPUT_INDEX]);


#if LOG_ENABLE
	fprintf(stderr, "SpatialConvolutionMM_MKLDNN_bwdData: start. \n");
	fprintf(stderr, "	input->size[0]=%d,input->size[1]=%d,input->size[2]=%d,input->size[3]=%d \n", input->size[0],input->size[1],input->size[2],input->size[3]);	
	fprintf(stderr, "	output->size[0]=%d,output->size[1]=%d,output->size[2]=%d,output->size[3]=%d \n", gradOutput->size[0],gradOutput->size[1],gradOutput->size[2],gradOutput->size[3]);
	fprintf(stderr, "	weight->size[0]=%d,weight->size[1]=%d\n", weight->size[0],weight->size[1]);
#endif
	THTensor_(resizeAs)(gradInput, input);
	THTensor_(zero)(gradInput);

	real * inPtr = THTensor_(data)(gradInput);
	real * filterPtr = THTensor_(data)(weight);
	real * outPtr = THTensor_(data)(gradOutput);
	//real * biasPtr = THTensor_(data)(bias);

	real * resConv[dnnResourceNumber]={0};
	resConv[dnnResourceDiffSrc] = inPtr;
	resConv[dnnResourceFilter] = filterPtr;
	resConv[dnnResourceDiffDst] = outPtr;
	//resConv[dnnResourceDiffBias] = biasPtr;

/*
	struct timeval check1;
	gettimeofday(&check1,NULL);
	double duration1 = (check1.tv_sec - start.tv_sec) * 1000 + (double)(check1.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	bwdData MKLDNN check1 time = %.2f ms\n",duration1 );
*/	//fprintf(stderr, "	m_conv = 0x%x, inPtr=0x%x, filterPtr=0x%x, outPtr=0x%x\n", m_conv_bwdData,inPtr,filterPtr,outPtr);
	

	if(sizeof(real) == sizeof(float))
	{
		//fprintf(stderr, "		cv1=0x%x, cv2=0x%x, cv3=0x%x\n",cv_user_to_conv1_filt,cv_user_to_conv1_output,cv_conv_to_user_input);
		//fprintf(stderr, "		buf1=0x%x, buf2=0x%x, buf3=0x%x\n",newInPtr,newFilterPtr,newOutPtr);
		if(cv_user_to_conv1_output){
			resConv[dnnResourceDiffDst] = newOutPtr;
			CHECK_ERR( dnnConversionExecute_F32(cv_user_to_conv1_output, outPtr, resConv[dnnResourceDiffDst]), err );
			//fprintf(stderr,"	conversion 1 ok.\n" );
		} 
		else{
			resConv[dnnResourceDiffDst] = outPtr;
		}
		
		if(cv_user_to_conv1_filt){
			resConv[dnnResourceFilter] = newFilterPtr;
			CHECK_ERR( dnnConversionExecute_F32(cv_user_to_conv1_filt, filterPtr, resConv[dnnResourceFilter]), err );
			//fprintf(stderr,"	conversion 2 ok.\n" );
		} 
		else{
			resConv[dnnResourceFilter] = filterPtr;
		}
		if(cv_conv_to_user_input){
			resConv[dnnResourceDiffSrc] = newInPtr;
		}
		else{
			resConv[dnnResourceDiffSrc] = inPtr;
		}

		//fprintf(stderr, "		call float api:dnnExecute_F32 start . \n");

		CHECK_ERR(dnnExecute_F32(m_conv_bwdData, (void**)resConv),err);	
		if(cv_conv_to_user_input){
			CHECK_ERR( dnnConversionExecute_F32(cv_conv_to_user_input, newInPtr, inPtr), err );
			//fprintf(stderr,"	conversion 3 ok.\n" );
		}
	}
	else if(sizeof(real) == sizeof(double))
	{
		CHECK_ERR(dnnExecute_F64(m_conv_bwdData, (void**)resConv),err);
	}
	gettimeofday(&end,NULL);
	double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
#if LOG_ENABLE
	fprintf(stderr,"	bwdData MKLDNN time = %.2f ms\n",duration );
#endif
}



void THNN_(SpatialConvolutionMM_MKLDNN_bwdFilter)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          THLongTensor *primitives,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          real scale)
{

	struct timeval start,end;
	gettimeofday(&start,NULL);
	dnnError_t err;
	dnnPrimitive_t m_conv_bwdFilter =NULL; 
	dnnPrimitive_t cv_user_to_conv1_input = NULL,
		cv_user_to_conv1_output = NULL,
		cv_conv_to_user_filt = NULL;
	real * newInPtr = NULL;
	real * newFilterPtr = NULL;
	real * newBiasPtr = NULL;
	real * newOutPtr = NULL;

	m_conv_bwdFilter = (dnnPrimitive_t) (primitives->storage->data[BWD_FILTER_INDEX]);
	cv_user_to_conv1_input 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_INPUT_INDEX];
	cv_user_to_conv1_output  = (dnnPrimitive_t)primitives->storage->data[CONVERT_CONV_OUTPUT_INDEX];
	cv_conv_to_user_filt 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDFILTER_INDEX];

	newInPtr 		= (real *)(primitives->storage->data[BUF_CONVERT_INPUT_INDEX]);
	newFilterPtr 		= (real *)(primitives->storage->data[BUF_CONVERT_FILTER_INDEX]);
	newOutPtr 		= (real *)(primitives->storage->data[BUF_CONVERT_OUTPUT_INDEX]);

#if LOG_ENABLE
	fprintf(stderr, "SpatialConvolutionMM_MKLDNN_bwdFilter: start. \n");
	fprintf(stderr, "	input->nDimension = %d, finput->nDimension = %d, gradWeight->nDimension = %d\n", input->nDimension,finput->nDimension,gradWeight->nDimension);
	fprintf(stderr, "	input->size[0]=%d,input->size[1]=%d,input->size[2]=%d,input->size[3]=%d \n", input->size[0],input->size[1],input->size[2],input->size[3]);	
	fprintf(stderr, "	output->size[0]=%d,output->size[1]=%d,output->size[2]=%d,output->size[3]=%d \n", gradOutput->size[0],gradOutput->size[1],gradOutput->size[2],gradOutput->size[3]);
	fprintf(stderr, "	weight->size[0]=%d,weight->size[1]=%d\n", gradWeight->size[0],gradWeight->size[1]);
#endif

	real * inPtr = THTensor_(data)(input);
	real * filterPtr = THTensor_(data)(gradWeight);
	real * outPtr = THTensor_(data)(gradOutput);
    	real * biasPtr = THTensor_(data)(gradBias);

	real * resConv[dnnResourceNumber]={0};
	resConv[dnnResourceSrc] = inPtr;
	resConv[dnnResourceDiffFilter] = filterPtr;
	resConv[dnnResourceDiffDst] = outPtr;
	resConv[dnnResourceDiffBias] = biasPtr;


	//fprintf(stderr, "	m_conv = 0x%x, inPtr=0x%x, filterPtr=0x%x, outPtr=0x%x\n", m_conv_bwdFilter,inPtr,filterPtr,outPtr);

	if(sizeof(real) == sizeof(float))
	{	
		//fprintf(stderr, "		cv1=0x%x, cv2=0x%x, cv3=0x%x\n",cv_user_to_conv1_input,cv_conv_to_user_filt,cv_user_to_conv1_output);
		//fprintf(stderr, "		buf1=0x%x, buf2=0x%x, buf3=0x%x\n",newInPtr,newFilterPtr,newOutPtr);

		if(cv_user_to_conv1_input){
			resConv[dnnResourceSrc] = newInPtr;
			CHECK_ERR( dnnConversionExecute_F32(cv_user_to_conv1_input, inPtr, resConv[dnnResourceSrc]), err );
		} 
		else{
			resConv[dnnResourceSrc] = inPtr;
		}
		if(cv_user_to_conv1_output){
			resConv[dnnResourceDiffDst] = newOutPtr;
			CHECK_ERR( dnnConversionExecute_F32(cv_user_to_conv1_output, outPtr, resConv[dnnResourceDiffDst]), err );
		} 
		else{
			resConv[dnnResourceDiffDst] = outPtr;
		}
		if(cv_conv_to_user_filt){
			resConv[dnnResourceDiffFilter] = newFilterPtr;
		}
		else{
			resConv[dnnResourceDiffFilter] = filterPtr;
		}


		CHECK_ERR(dnnExecute_F32(m_conv_bwdFilter, (void**)resConv),err);
		if(cv_conv_to_user_filt){
			CHECK_ERR( dnnConversionExecute_F32(cv_conv_to_user_filt, newFilterPtr, filterPtr), err );
		}
		//fprintf(stderr, "		call float api:dnnExecute_F32 end, out[0]=%.2f \n",outPtr[0]);
	}

	else if(sizeof(real) == sizeof(double))
	{
		CHECK_ERR(dnnExecute_F64(m_conv_bwdFilter, (void**)resConv),err);
	}
#if LOG_ENABLE
	gettimeofday(&end,NULL);
	double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	bwdFilter MKLDNN time = %.2f ms\n",duration );
#endif
}


static void THNN_(SpatialConvolutionMM_updateOutput_frame)(
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          long nInputPlane,
          long inputWidth,
          long inputHeight,
          long nOutputPlane,
          long outputWidth,
          long outputHeight)
{
  long i;
  THTensor *output2d;

  THNN_(unfolded_copy)(finput, input, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);

  output2d = THTensor_(newWithStorage2d)(output->storage, output->storageOffset,
                                         nOutputPlane, -1,
                                         outputHeight*outputWidth, -1);
  if (bias) {
    for(i = 0; i < nOutputPlane; i++)
        THVector_(fill)(output->storage->data+output->storageOffset+output->stride[0]*i, THTensor_(get1d)(bias, i), outputHeight*outputWidth);
  } else {
    THTensor_(zero)(output);
  }

  THTensor_(addmm)(output2d, 1, output2d, 1, weight, finput);

  THTensor_(free)(output2d);
}

void THNN_(SpatialConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
  int dimf = 0;
  int dimw = 2;
  int dimh = 1;

  long nInputPlane;
  long inputWidth;
  long inputHeight;
  long nOutputPlane;
  long outputWidth;
  long outputHeight;

  THArgCheck( input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor expected");
  THArgCheck(kW > 0 && kH > 0, 8, "kernel size should be greater than zero");
  THArgCheck(dW > 0 && dH > 0, 10, "stride should be greater than zero");

  if (input->nDimension == 4) {
    dimf++;
    dimw++;
    dimh++;
  }

  nInputPlane = input->size[dimf];
  inputWidth   = input->size[dimw];
  inputHeight  = input->size[dimh];
  nOutputPlane = weight->size[0];
  outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
        nInputPlane,inputHeight,inputWidth,nOutputPlane,outputHeight,outputWidth);

  if (nInputPlane*kW*kH != weight->size[1])
    THError("Wrong number of input channels! Input has %d channels, expected %d",nInputPlane,weight->size[1]/(kW*kH));

  if(input->nDimension == 3)
  {
    THTensor_(resize2d)(finput, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);

    THNN_(SpatialConvolutionMM_updateOutput_frame)(input, output, weight, bias, finput,
                                                 kW, kH, dW, dH, padW, padH,
                                                 nInputPlane, inputWidth, inputHeight,
                                                 nOutputPlane, outputWidth, outputHeight);
  }
  else
  {
    long T = input->size[0];
    long t;

    THTensor_(resize3d)(finput, T, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize4d)(output, T, nOutputPlane, outputHeight, outputWidth);

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *input_t = THTensor_(newSelect)(input, 0, t);
      THTensor *output_t = THTensor_(newSelect)(output, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      THNN_(SpatialConvolutionMM_updateOutput_frame)(input_t, output_t, weight, bias, finput_t,
                                                   kW, kH, dW, dH, padW, padH,
                                                   nInputPlane, inputWidth, inputHeight,
                                                   nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      THTensor_(free)(finput_t);
    }
  }
}

static void THNN_(SpatialConvolutionMM_updateGradInput_frame)(
          THTensor *gradInput,
          THTensor *gradOutput,
          THTensor *weight,
          THTensor *fgradInput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, gradOutput->storageOffset,
                                                       gradOutput->size[0], -1,
                                                       gradOutput->size[1]*gradOutput->size[2], -1);
  THTensor_(addmm)(fgradInput, 0, fgradInput, 1, weight, gradOutput2d);
  THTensor_(free)(gradOutput2d);

  THTensor_(zero)(gradInput);

  THNN_(unfolded_acc)(fgradInput, gradInput, kW, kH, dW, dH, padW, padH, gradInput->size[0], gradInput->size[2], gradInput->size[1], gradOutput->size[2], gradOutput->size[1]);
}

void THNN_(SpatialConvolutionMM_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
  long nOutputPlane = weight->size[0];

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 3, "Number of output features is not equal to nOutputPlane" );
  THArgCheck(kW > 0 && kH > 0, 9, "kernel size should be greater than zero");
  THArgCheck(dW > 0 && dH > 0, 11, "stride should be greater than zero");

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(resizeAs)(fgradInput, finput);
  THTensor_(transpose)(weight, weight, 0, 1);

  if(input->nDimension == 3)
  {
    THNN_(SpatialConvolutionMM_updateGradInput_frame)(gradInput, gradOutput, weight, fgradInput, kW, kH, dW, dH, padW, padH);
  }
  else
  {
    long T = input->size[0];
    long t;

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *fgradInput_t = THTensor_(newSelect)(fgradInput, 0, t);

      THNN_(SpatialConvolutionMM_updateGradInput_frame)(gradInput_t, gradOutput_t, weight, fgradInput_t, kW, kH, dW, dH, padW, padH);

      THTensor_(free)(gradInput_t);
      THTensor_(free)(gradOutput_t);
      THTensor_(free)(fgradInput_t);
    }
  }

  THTensor_(transpose)(weight, weight, 0, 1);
}

static void THNN_(SpatialConvolutionMM_accGradParameters_frame)(
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          real scale)
{
  long i;
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, gradOutput->storageOffset,
                                                       gradOutput->size[0], -1,
                                                       gradOutput->size[1]*gradOutput->size[2], -1);

  THTensor_(transpose)(finput, finput, 0, 1);
  THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutput2d, finput);
  THTensor_(transpose)(finput, finput, 0, 1);

  if (gradBias) {
    for(i = 0; i < gradBias->size[0]; i++)
    {
      long k;
      real sum = 0;
      real *data = gradOutput2d->storage->data + gradOutput2d->storageOffset + i*gradOutput2d->stride[0];
      for(k = 0; k < gradOutput2d->size[1]; k++)
        sum += data[k];
      (gradBias->storage->data + gradBias->storageOffset)[i] += scale*sum;
    }
  }

  THTensor_(free)(gradOutput2d);
}

void THNN_(SpatialConvolutionMM_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          real scale)
{
  long nOutputPlane = gradWeight->size[0];
  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 3, "Number of output features is not equal to nOutputPlane" );
  THArgCheck(kW > 0 && kH > 0, 8, "kernel size should be greater than zero");
  THArgCheck(dW > 0 && dH > 0, 10, "stride should be greater than zero");

  if(input->nDimension == 3)
  {
    THNN_(SpatialConvolutionMM_accGradParameters_frame)(gradOutput, gradWeight, gradBias, finput, scale);
  }
  else
  {
    long T = input->size[0];
    long t;

    for(t = 0; t < T; t++)
    {
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      THNN_(SpatialConvolutionMM_accGradParameters_frame)(gradOutput_t, gradWeight, gradBias, finput_t, scale);

      THTensor_(free)(gradOutput_t);
      THTensor_(free)(finput_t);
    }
  }
}

#endif
