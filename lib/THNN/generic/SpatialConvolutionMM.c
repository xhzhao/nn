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
    int bZeros = 1;
    real threshold = 0.0001;
    if(compareSource == 2 || compareSource == 3 || compareSource == 5 || compareSource == 7)
    {   
        threshold = 0.00001;
    }   

    for(i=0; i< len; i++)
    {   
        if( (ptr1[i]-ptr2[i] > threshold ) || (ptr1[i]-ptr2[i] < (0-threshold) ))
        {
            break;
        }
        if(ptr1[i] > threshold || ptr1[i] < (0-threshold))
        {
            bZeros = 0;
        }
    }   

    if(i == len)
    {   
//#if LOG_ENABLE
        fprintf(stderr, "   compareSource = %d, mkldnn is same as old, good. len =%ld, bZeros=%d, mkldnn[0]=%.4f,mkldnn[1]=%.4f,mkldnn[2]=%.4f, mkldnn[15]=%.4f, old[15]=%.4f\n",
            compareSource,len,bZeros,ptr1[0],ptr1[1],ptr1[2], ptr1[15], ptr2[15]);
//#endif    
    }   
    else
    {   
        fprintf(stderr, "   compareSource = %d, compare fail: i =%d, len=%d, mkldnn[i] = %.5f, old[i]=%.5f ,mkldnn/old=%.4f \n", compareSource, i, len, ptr1[i], ptr2[i],ptr1[i]/ptr2[i]);
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



	size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };
	size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
	size_t filterStrides[dimension] = { 1, kW, kH * kW, inC * kH * kW };

	size_t biasSize[1] = { outputSize[2] };
	size_t biasStrides[1] = { 1 };
	//size_t biasStrides[1] = { outputStrides[2] };

	//user layouts
	dnnLayout_t lt_user_input, lt_user_filter, lt_user_bias, lt_user_output;
	//forward layouts
	dnnLayout_t lt_forward_conv_input, lt_forward_conv_filter, lt_forward_conv_bias, lt_forward_conv_output;
	//backward data layouts
	dnnLayout_t lt_bwddata_conv_input, lt_bwddata_conv_filter,lt_bwddata_conv_output;
	//backward filter layouts
	dnnLayout_t lt_bwdfilter_conv_input, lt_bwdfilter_conv_filter,lt_bwdfilter_conv_output;

	//forward conversions and buffers
	dnnPrimitive_t cv_forward_input = NULL,cv_forward_filter = NULL,cv_forward_bias = NULL,cv_forward_output = NULL;
	real * buffer_forward_input =NULL;real *buffer_forward_filter=NULL;real *buffer_forward_bias=NULL;real * buffer_forward_output =NULL;
	//backward data conversions and buffers
	dnnPrimitive_t cv_bwddata_input = NULL,cv_bwddata_filter = NULL,cv_bwddata_output = NULL;
	real * buffer_bwddata_input = NULL;real * buffer_bwddata_filter = NULL;real * buffer_bwddata_output=NULL;
	//backward filter conversions and buffers
	dnnPrimitive_t cv_bwdfilter_input = NULL,cv_bwdfilter_filter = NULL,cv_bwdfilter_output = NULL;
	real * buffer_bwdfilter_input = NULL;real * buffer_bwdfilter_filter = NULL;real * buffer_bwdfilter_output = NULL;


#if NEW_INTERFACE
/*for new interface*/
	dnnPrimitiveAttributes_t attributes = NULL;
	CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );
#endif
/**/


	if(sizeof(real) == sizeof(float))
	{
		if(primitives->storage->data[CONV_LAYOUT_INPUT] == 0)
		{
			CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides) , err );
			fprintf(stderr ,"MKLDNN Convolution fail to get input layout \n");
		}
		else{
			lt_user_input = primitives->storage->data[CONV_LAYOUT_INPUT];
			fprintf(stderr ,"MKLDNN Convolution get valid input layout \n");
		}
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_filter, dimension, filterSize, filterStrides), err );
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

		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_input, m_conv_forward, dnnResourceSrc) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_filter, m_conv_forward, dnnResourceFilter), err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_bias, m_conv_forward, dnnResourceBias) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_output,m_conv_forward, dnnResourceDst) , err );	

		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwddata_conv_input, m_conv_bwd_data, dnnResourceDiffSrc) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwddata_conv_filter, m_conv_bwd_data, dnnResourceFilter) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwddata_conv_output, m_conv_bwd_data, dnnResourceDiffDst) , err );

		int check1,check2,check3;
		check1 = dnnLayoutCompare_F32(lt_bwddata_conv_input, lt_user_input);
		check2 = dnnLayoutCompare_F32(lt_bwddata_conv_filter, lt_user_filter);
		check3 = dnnLayoutCompare_F32(lt_bwddata_conv_output, lt_user_output);
		//fprintf(stderr, "	check1=%d, check2=%d, check3=%d",check1,check2,check3);

		int check4,check5,check6;
		check4 = dnnLayoutCompare_F32(lt_bwddata_conv_input, lt_forward_conv_input);
		check5 = dnnLayoutCompare_F32(lt_bwddata_conv_filter, lt_forward_conv_filter);
		check6 = dnnLayoutCompare_F32(lt_bwddata_conv_output, lt_forward_conv_output);
		//fprintf(stderr, "	check4=%d, check5=%d, check6=%d \n",check4,check5,check6);


		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwdfilter_conv_input, m_conv_bwd_filter, dnnResourceSrc) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwdfilter_conv_filter, m_conv_bwd_filter, dnnResourceDiffFilter) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwdfilter_conv_output, m_conv_bwd_filter, dnnResourceDiffDst) , err );

		//init forward conversions:
		CHECK_ERR( THNN_(init_conversion)(&cv_forward_input, 	&buffer_forward_input, 	lt_forward_conv_input, 	lt_user_input) , err );
		CHECK_ERR( THNN_(init_conversion)(&cv_forward_filter, 	&buffer_forward_filter, lt_forward_conv_filter, lt_user_filter), err );
		CHECK_ERR( THNN_(init_conversion)(&cv_forward_bias, 	&buffer_forward_bias, 	lt_forward_conv_bias, 	lt_user_bias), err );
		if(!dnnLayoutCompare_F32(lt_forward_conv_output, lt_user_output))
		{
			CHECK_ERR( dnnConversionCreate_F32(&cv_forward_output, 	lt_forward_conv_output, lt_user_output), err );
			CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forward_output), lt_forward_conv_output), err );
		}

		//init backward data conversion:
		if(!dnnLayoutCompare_F32(lt_bwddata_conv_input, lt_user_input))
		{
			CHECK_ERR( dnnConversionCreate_F32(&cv_bwddata_input, lt_bwddata_conv_input, lt_user_input), err );

			if(dnnLayoutCompare_F32(lt_forward_conv_input, lt_bwddata_conv_input))
			{
				buffer_bwddata_input = buffer_forward_input;
			}
			else
			{
				CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_bwddata_input), lt_bwddata_conv_input), err );
				fprintf(stderr, "bwddata init 1\n");
			}
		}

		if(dnnLayoutCompare_F32(lt_forward_conv_filter, lt_bwddata_conv_filter))
		{
			cv_bwddata_filter = cv_forward_filter;
			buffer_bwddata_filter = buffer_forward_filter;
			fprintf(stderr, "bwdfilter init 2\n");
		}
		else
		{
			CHECK_ERR( THNN_(init_conversion)(&cv_bwddata_filter, 	&buffer_bwddata_filter, lt_bwddata_conv_filter, lt_user_filter) , err );
		}

		CHECK_ERR( THNN_(init_conversion)(&cv_bwddata_output, 	&buffer_bwddata_output, lt_bwddata_conv_output, lt_user_output) , err );

		//init backward filter conversions:
		if(dnnLayoutCompare_F32(lt_forward_conv_input, lt_bwdfilter_conv_input))
		{
			cv_bwdfilter_input = cv_forward_input;
			buffer_bwdfilter_input = buffer_forward_input;
		}
		else
		{
			CHECK_ERR( THNN_(init_conversion)(&cv_bwdfilter_input, &buffer_bwdfilter_input, lt_bwdfilter_conv_input, lt_user_input) , err );
		}
		if(!dnnLayoutCompare_F32(lt_bwdfilter_conv_filter, lt_user_filter))
		{
			CHECK_ERR( dnnConversionCreate_F32(&cv_bwdfilter_filter, lt_bwdfilter_conv_filter, lt_user_filter), err );
			if(dnnLayoutCompare_F32(lt_forward_conv_filter, lt_bwdfilter_conv_filter))
			{
				buffer_bwdfilter_filter = buffer_forward_filter;
			}
			else
			{
				CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_bwdfilter_filter), lt_bwdfilter_conv_filter), err );
			}
		}

		if(dnnLayoutCompare_F32(lt_bwddata_conv_output, lt_bwdfilter_conv_output))
		{
			//fprintf(stderr, "bwdfilter init 3");
			cv_bwdfilter_output = cv_bwddata_output;
			buffer_bwdfilter_output = buffer_bwddata_output;
		}
		else 

		{
			CHECK_ERR( THNN_(init_conversion)(&cv_bwdfilter_output, &buffer_bwdfilter_output, lt_bwdfilter_conv_output, lt_user_output) , err );
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
	//save the output layout to dnnPrimitive
	primitives->storage->data[CONV_LAYOUT_OUTPUT] = (long long)lt_forward_conv_output;
	primitives->storage->data[FORWARD_INDEX] = (long long)m_conv_forward;
	primitives->storage->data[BWD_DATA_INDEX] = (long long)m_conv_bwd_data;
	primitives->storage->data[BWD_FILTER_INDEX] = (long long)m_conv_bwd_filter;

	primitives->storage->data[CONVERT_FORWARD_INPUT] 	= (long long)cv_forward_input;
	primitives->storage->data[CONVERT_FORWARD_FILTER] 	= (long long)cv_forward_filter;
	primitives->storage->data[CONVERT_FORWARD_BIAS] 	= (long long)cv_forward_bias;
	primitives->storage->data[CONVERT_FORWARD_OUTPUT] 	= (long long)cv_forward_output;

	primitives->storage->data[CONVERT_BWDDATA_INPUT] 	= (long long)cv_bwddata_input;
	primitives->storage->data[CONVERT_BWDDATA_FILTER] 	= (long long)cv_bwddata_filter;
	primitives->storage->data[CONVERT_BWDDATA_OUTPUT] 	= (long long)cv_bwddata_output;

	primitives->storage->data[CONVERT_BWDFILTER_INPUT] 	= (long long)cv_bwdfilter_input;
	primitives->storage->data[CONVERT_BWDFILTER_FILTER] 	= (long long)cv_bwdfilter_filter;
	primitives->storage->data[CONVERT_BWDFILTER_OUTPUT] 	= (long long)cv_bwdfilter_output;

	primitives->storage->data[BUFFER_FORWARD_INPUT] 	= (long long)buffer_forward_input;
	primitives->storage->data[BUFFER_FORWARD_FILTER] 	= (long long)buffer_forward_filter;
	primitives->storage->data[BUFFER_FORWARD_BIAS] 		= (long long)buffer_forward_bias;
	primitives->storage->data[BUFFER_FORWARD_OUTPUT] 	= (long long)buffer_forward_output;

	primitives->storage->data[BUFFER_BWDDATA_INPUT] 	= (long long)buffer_bwddata_input;
	primitives->storage->data[BUFFER_BWDDATA_FILTER] 	= (long long)buffer_bwddata_filter;
	primitives->storage->data[BUFFER_BWDDATA_OUTPUT] 	= (long long)buffer_bwddata_output;

	primitives->storage->data[BUFFER_BWDFILTER_INPUT] 	= (long long)buffer_bwdfilter_input;
	primitives->storage->data[BUFFER_BWDFILTER_FILTER] 	= (long long)buffer_bwdfilter_filter;
	primitives->storage->data[BUFFER_BWDFILTER_OUTPUT] 	= (long long)buffer_bwdfilter_output;



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
	struct timeval start,mid,convert1,convert2,end;
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

	dnnPrimitive_t cv_forward_input = NULL,cv_forward_filter = NULL,cv_forward_bias = NULL,cv_forward_output = NULL;
	real * buffer_forward_input =NULL;real *buffer_forward_filter=NULL;real *buffer_forward_bias=NULL;real * buffer_forward_output =NULL;

	if(initOk == 0)
	{
		THNN_(SpatialConvolutionMM_MKLDNN_init)(primitives,N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW);
	}
	m_conv_forward 		= (dnnPrimitive_t)(primitives->storage->data[FORWARD_INDEX]);
	cv_forward_input 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_FORWARD_INPUT];
	cv_forward_filter 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_FORWARD_FILTER];
	cv_forward_bias 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_FORWARD_BIAS];
	cv_forward_output  	= (dnnPrimitive_t)primitives->storage->data[CONVERT_FORWARD_OUTPUT];
	buffer_forward_input 	= (real *)(primitives->storage->data[BUFFER_FORWARD_INPUT]);
	buffer_forward_filter 	= (real *)(primitives->storage->data[BUFFER_FORWARD_FILTER]);
	buffer_forward_bias 	= (real *)(primitives->storage->data[BUFFER_FORWARD_BIAS]);
	buffer_forward_output 	= (real *)(primitives->storage->data[BUFFER_FORWARD_OUTPUT]);


	THTensor_(resize3d)(finput, N, kW*kH*inC, outH*outW);
	THTensor_(resize4d)(output, N, outC, outH, outW);
	//return;
#if LOG_ENABLE
	gettimeofday(&mid,NULL);
	fprintf(stderr, "SpatialConvolutionMM_MKLDNN_forward: start, m_conv_forward = 0x%x \n",m_conv_forward);
	fprintf(stderr, "	input->size[0]=%d,input->size[1]=%d,input->size[2]=%d,input->size[3]=%d \n", input->size[0],input->size[1],input->size[2],input->size[3]);	
	fprintf(stderr, "	output->size[0]=%d,output->size[1]=%d,output->size[2]=%d,output->size[3]=%d \n", output->size[0],output->size[1],output->size[2],output->size[3]);
	fprintf(stderr, "	weight->size[0]=%d,weight->size[1]=%d\n", weight->size[0],weight->size[1]);
	fprintf(stderr, "	bias->nDimension=%d,bias->size[0]=%d,bias->storage->data[0]=%.3f\n", bias->nDimension,bias->size[0],bias->storage->data[0]);
	fprintf(stderr, " cv_forward_input=0x%x,cv_forward_filter=0x%x,cv_forward_bias=0x%x,cv_forward_output=0x%x",cv_forward_input,cv_forward_filter,cv_forward_bias,cv_forward_output);
#endif

/**/
	long long i = 0;
	real * inPtr = THTensor_(data)(input);
	real * filterPtr = THTensor_(data)(weight);
	real * outPtr = THTensor_(data)(output);
	real * biasPtr = THTensor_(data)(bias);


	real * resConv[dnnResourceNumber]={0};
	resConv[dnnResourceSrc] 	= inPtr;
	resConv[dnnResourceFilter] 	= filterPtr;
	resConv[dnnResourceBias] 	= biasPtr;
	resConv[dnnResourceDst] 	= outPtr;

	void *convert_resources[dnnResourceNumber];

	if(sizeof(real) == sizeof(float))
	{
		if(cv_forward_input){
			resConv[dnnResourceSrc] = buffer_forward_input;
			convert_resources[dnnResourceFrom] = inPtr;
			convert_resources[dnnResourceTo]   = buffer_forward_input;
			CHECK_ERR( dnnExecute_F32(cv_forward_input, convert_resources), err );
			fprintf(stderr, "SpatialConvolutionMM_MKLDNN_forward conversion input \n");
		}
		
		if(cv_forward_filter){
			resConv[dnnResourceFilter] = buffer_forward_filter;
			convert_resources[dnnResourceFrom] = filterPtr;
			convert_resources[dnnResourceTo]   = buffer_forward_filter;
			CHECK_ERR( dnnExecute_F32(cv_forward_filter, convert_resources), err );
			fprintf(stderr, "SpatialConvolutionMM_MKLDNN_forward conversion filter \n");
		} 
		
		if(cv_forward_bias){
			resConv[dnnResourceBias] = buffer_forward_bias;
			convert_resources[dnnResourceFrom] = biasPtr;
			convert_resources[dnnResourceTo]   = buffer_forward_bias;
			CHECK_ERR( dnnExecute_F32(cv_forward_bias,convert_resources), err );
			fprintf(stderr, "SpatialConvolutionMM_MKLDNN_forward conversion bias \n");
		} 

		if(cv_forward_output){
			resConv[dnnResourceDst] = buffer_forward_output;
		} 
		gettimeofday(&convert1,NULL);

		CHECK_ERR(dnnExecute_F32(m_conv_forward, (void**)resConv),err);	
		gettimeofday(&convert2,NULL);

		if(cv_forward_output){

/*			convert_resources[dnnResourceFrom] = buffer_forward_output;
			convert_resources[dnnResourceTo]   = outPtr;
			CHECK_ERR( dnnExecute_F32(cv_forward_output, convert_resources), err );
*/
			//release the original buffer, replace it with the internal buffer
			output->storage->data = buffer_forward_output;
			
			
		} 
		//fprintf(stderr, "		call float api:dnnExecute_F32 end, out[0]=%.2f \n",outPtr[0]);
	}
	else if(sizeof(real) == sizeof(double))
	{

		CHECK_ERR(dnnExecute_F64(m_conv_forward, (void**)resConv),err);
	}
#if LOG_ENABLE
	gettimeofday(&end,NULL);
	double duration1 = (mid.tv_sec - start.tv_sec) * 1000 + (double)(mid.tv_usec - start.tv_usec) /1000;
	double duration2 = (end.tv_sec - mid.tv_sec) * 1000 + (double)(end.tv_usec - mid.tv_usec) /1000;
	fprintf(stderr,"	forward MKLDNN time1 = %.2f ms, time2 = %.2f\nms",duration1,duration2);
	double convert_time1 = (convert1.tv_sec - mid.tv_sec) * 1000 + (double)(convert1.tv_usec - mid.tv_usec) /1000;
	double exec_time = (convert2.tv_sec - convert1.tv_sec) * 1000 + (double)(convert2.tv_usec - convert1.tv_usec) /1000;
	double convert_time2 = (end.tv_sec - convert2.tv_sec) * 1000 + (double)(end.tv_usec - convert2.tv_usec) /1000;
	fprintf(stderr,"	forward MKLDNN convert_time1 = %.2f ms, exec_time = %.2f, convert_time2=%.2f\nms",convert_time1,exec_time,convert_time2);
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
	struct timeval start,mid1,mid2,mid3,convert1,convert2,end;
	gettimeofday(&start,NULL);
	dnnError_t err;
	dnnPrimitive_t m_conv_bwdData =NULL; 

	dnnPrimitive_t cv_bwddata_input = NULL,cv_bwddata_filter = NULL,cv_bwddata_output = NULL;
	real * buffer_bwddata_input = NULL;real * buffer_bwddata_filter = NULL;real * buffer_bwddata_output=NULL;

	THTensor_(transpose)(weight, weight, 0, 1);

	gettimeofday(&mid1,NULL);

	m_conv_bwdData = (dnnPrimitive_t) (primitives->storage->data[BWD_DATA_INDEX]);
	cv_bwddata_input 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDDATA_INPUT];
	cv_bwddata_filter 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDDATA_FILTER];
	cv_bwddata_output  	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDDATA_OUTPUT];
	buffer_bwddata_input 	= (real *)(primitives->storage->data[BUFFER_BWDDATA_INPUT]);
	buffer_bwddata_filter 	= (real *)(primitives->storage->data[BUFFER_BWDDATA_FILTER]);
	buffer_bwddata_output 	= (real *)(primitives->storage->data[BUFFER_BWDDATA_OUTPUT]);




	THTensor_(resizeAs)(gradInput, input);
	gettimeofday(&mid2,NULL);
	THTensor_(zero)(gradInput);


#if LOG_ENABLE
	gettimeofday(&mid3,NULL);
	fprintf(stderr, "SpatialConvolutionMM_MKLDNN_bwdData: start. \n");
	fprintf(stderr, "	input->size[0]=%d,input->size[1]=%d,input->size[2]=%d,input->size[3]=%d \n", input->size[0],input->size[1],input->size[2],input->size[3]);	
	fprintf(stderr, "	output->size[0]=%d,output->size[1]=%d,output->size[2]=%d,output->size[3]=%d \n", gradOutput->size[0],gradOutput->size[1],gradOutput->size[2],gradOutput->size[3]);
	fprintf(stderr, "	weight->size[0]=%d,weight->size[1]=%d\n", weight->size[0],weight->size[1]);
#endif

	real * inPtr = THTensor_(data)(gradInput);
	real * filterPtr = THTensor_(data)(weight);
	real * outPtr = THTensor_(data)(gradOutput);

	real * resConv[dnnResourceNumber]={0};
	resConv[dnnResourceDiffSrc] = inPtr;
	resConv[dnnResourceFilter] = filterPtr;
	resConv[dnnResourceDiffDst] = outPtr;

	void *convert_resources[dnnResourceNumber];
	if(sizeof(real) == sizeof(float))
	{
		if(cv_bwddata_output){
			resConv[dnnResourceDiffDst] = buffer_bwddata_output;
			convert_resources[dnnResourceFrom] = outPtr;
			convert_resources[dnnResourceTo]   = buffer_bwddata_output;
			//CHECK_ERR( dnnExecute_F32(cv_bwddata_output,convert_resources), err );
			//fprintf(stderr, "		convert 1 called. \n");
		}
		
		if(cv_bwddata_filter){
			resConv[dnnResourceFilter] = buffer_bwddata_filter;
			convert_resources[dnnResourceFrom] = filterPtr;
			convert_resources[dnnResourceTo]   = buffer_bwddata_filter;
			//CHECK_ERR( dnnExecute_F32(cv_bwddata_filter, convert_resources), err );
			//fprintf(stderr, "		convert 2 called. \n");
		}

		if(cv_bwddata_input){
			resConv[dnnResourceDiffSrc] = buffer_bwddata_input;;
		}

		
		gettimeofday(&convert1,NULL);
		CHECK_ERR(dnnExecute_F32(m_conv_bwdData, (void**)resConv),err);	
		gettimeofday(&convert2,NULL);

		if(cv_bwddata_input){
			convert_resources[dnnResourceFrom] = buffer_bwddata_input;
			convert_resources[dnnResourceTo]   = inPtr;
			CHECK_ERR( dnnExecute_F32(cv_bwddata_input, convert_resources), err );
			//fprintf(stderr, "		convert 3 called. \n");
		}

	}
	else if(sizeof(real) == sizeof(double))
	{
		CHECK_ERR(dnnExecute_F64(m_conv_bwdData, (void**)resConv),err);
	}
	gettimeofday(&end,NULL);


#if LOG_ENABLE
	double time1 = (mid1.tv_sec - start.tv_sec) * 1000 + (double)(mid1.tv_usec - start.tv_usec) /1000;
	double time2 = (mid2.tv_sec - mid1.tv_sec) * 1000 + (double)(mid2.tv_usec - mid1.tv_usec) /1000;
	double time3 = (mid3.tv_sec - mid2.tv_sec) * 1000 + (double)(mid3.tv_usec - mid2.tv_usec) /1000;
	fprintf(stderr,"	bwdData MKLDNN mid1 = %.2f ms, mid2 = %.2f ms, mid3 = %.2f\n",time1,time2,time3) ;

	double duration1 = (mid3.tv_sec - start.tv_sec) * 1000 + (double)(mid3.tv_usec - start.tv_usec) /1000;
	double duration2 = (end.tv_sec - mid3.tv_sec) * 1000 + (double)(end.tv_usec - mid3.tv_usec) /1000;

	fprintf(stderr,"	bwdData MKLDNN time1 = %.2f ms, time2 = %.2f ms\n",duration1,duration2 );

        double convert_time1 = (convert1.tv_sec - mid3.tv_sec) * 1000 + (double)(convert1.tv_usec - mid3.tv_usec) /1000;
        double exec_time = (convert2.tv_sec - convert1.tv_sec) * 1000 + (double)(convert2.tv_usec - convert1.tv_usec) /1000;
        double convert_time2 = (end.tv_sec - convert2.tv_sec) * 1000 + (double)(end.tv_usec - convert2.tv_usec) /1000;
        fprintf(stderr,"        bwddata MKLDNN convert_time1 = %.2f ms, exec_time = %.2f, convert_time2=%.2f\nms",convert_time1,exec_time,convert_time2);


#endif
	THTensor_(transpose)(weight, weight, 0, 1);
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

	struct timeval start,mid,convert1,convert2,end;
	gettimeofday(&start,NULL);
	dnnError_t err;
	dnnPrimitive_t m_conv_bwdFilter =NULL; 
	dnnPrimitive_t cv_bwdfilter_input = NULL,cv_bwdfilter_filter = NULL,cv_bwdfilter_output = NULL;
	real * buffer_bwdfilter_input = NULL;real * buffer_bwdfilter_filter = NULL;real * buffer_bwdfilter_output = NULL;


	m_conv_bwdFilter = (dnnPrimitive_t) (primitives->storage->data[BWD_FILTER_INDEX]);
	cv_bwdfilter_input 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDFILTER_INPUT];
	cv_bwdfilter_filter 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDFILTER_FILTER];
	cv_bwdfilter_output  	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDFILTER_OUTPUT];
	buffer_bwdfilter_input 	= (real *)(primitives->storage->data[BUFFER_BWDFILTER_INPUT]);
	buffer_bwdfilter_filter = (real *)(primitives->storage->data[BUFFER_BWDFILTER_FILTER]);
	buffer_bwdfilter_output = (real *)(primitives->storage->data[BUFFER_BWDFILTER_OUTPUT]);




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
	void *convert_resources[dnnResourceNumber];

	//fprintf(stderr, "	m_conv = 0x%x, inPtr=0x%x, filterPtr=0x%x, outPtr=0x%x\n", m_conv_bwdFilter,inPtr,filterPtr,outPtr);

	if(sizeof(real) == sizeof(float))
	{	
		//fprintf(stderr, "		cv1=0x%x, cv2=0x%x, cv3=0x%x\n",cv_user_to_conv1_input,cv_conv_to_user_filt,cv_user_to_conv1_output);
		//fprintf(stderr, "		buf1=0x%x, buf2=0x%x, buf3=0x%x\n",newInPtr,newFilterPtr,newOutPtr);

		if(cv_bwdfilter_input){
			resConv[dnnResourceSrc] = buffer_bwdfilter_input;
			convert_resources[dnnResourceFrom] = inPtr;
			convert_resources[dnnResourceTo]   = buffer_bwdfilter_input;
			//CHECK_ERR( dnnExecute_F32(cv_bwdfilter_input, convert_resources), err );
		}
		if(cv_bwdfilter_output){
			resConv[dnnResourceDiffDst] = buffer_bwdfilter_output;
			convert_resources[dnnResourceFrom] = outPtr;
			convert_resources[dnnResourceTo]   = buffer_bwdfilter_output;
			//CHECK_ERR( dnnExecute_F32(cv_bwdfilter_output, convert_resources), err );
		}
		if(cv_bwdfilter_filter){
			resConv[dnnResourceDiffFilter] = buffer_bwdfilter_filter;
		}


		gettimeofday(&convert1,NULL);
		CHECK_ERR(dnnExecute_F32(m_conv_bwdFilter, (void**)resConv),err);
		gettimeofday(&convert2,NULL);
	
		if(cv_bwdfilter_filter){
			convert_resources[dnnResourceFrom] = buffer_bwdfilter_filter;
			convert_resources[dnnResourceTo]   = filterPtr;
			CHECK_ERR( dnnExecute_F32(cv_bwdfilter_filter, convert_resources), err );
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

        double convert_time1 = (convert1.tv_sec - start.tv_sec) * 1000 + (double)(convert1.tv_usec - start.tv_usec) /1000;
        double exec_time = (convert2.tv_sec - convert1.tv_sec) * 1000 + (double)(convert2.tv_usec - convert1.tv_usec) /1000;
        double convert_time2 = (end.tv_sec - convert2.tv_sec) * 1000 + (double)(end.tv_usec - convert2.tv_usec) /1000;
        fprintf(stderr,"        bwdfilter MKLDNN convert_time1 = %.2f ms, exec_time = %.2f, convert_time2=%.2f\nms",convert_time1,exec_time,convert_time2);
#

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
