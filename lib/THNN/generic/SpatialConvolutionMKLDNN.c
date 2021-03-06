#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionMKLDNN.c"
#else


#include "MKLDNN.h"

dnnError_t  THNN_(init_conversion)(dnnPrimitive_t *cv, real **ptr_out,
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


/*set the tensor pointer to newBuffer, and layout to newLayout*/
void THNN_(MKLDNN_set_tensor)(
          THNNState * state,
          THTensor * t,
          long long newBuffer,
	  long long newLayout
	)
{
	t->storage->data = (real * )newBuffer; //memory leak ??? need to check
	t->mkldnnLayout = newLayout;
	THStorage_(setMKLDNN)(t->storage);
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

/*Convert tensor from internal layout to NCHW layout
  The primitives save the conversions and buffers for each index.
  primitives[0] = conversion1
  primitives[1] = buffer1
  ...
*/
void THNN_(MKLDNN_ConvertLayoutBackToNCHW)(
          THNNState * state,
          THTensor * input,
          THLongTensor *primitives,
          int i,
          int initOk
	)
{
	dnnError_t err;
	dnnPrimitive_t cv_BacktoNCHW = NULL;
	real * dnnbuffer = NULL;
	real * torchbuffer = NULL;
	int N = input->size[0];
	int inC = input->size[1];
	int inH = input->size[2];
	int inW = input->size[3];
	dnnLayout_t mkldnnLayout = (dnnLayout_t)input->mkldnnLayout ;
#if LOG_ENABLE
	fprintf(stderr, "MKLDNN_ConvertLayoutBackToNCHW: start, N=%d,C=%d,H=%d,W=%d,mkldnnLayout = 0x%x, input=0x%x, THTensor_(data)(input) = 0x%x \n",N,inC,inH,inW,mkldnnLayout,input, THTensor_(data)(input));
#endif
	if(initOk == 0)
	{


		size_t inputSize[dimension] = 	{inW,inH,inC,N};
		size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
		dnnLayout_t lt_user_input = NULL;

		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides) , err );
#if CONVERSION_LOG
		int input_size = dnnLayoutGetMemorySize_F32(mkldnnLayout);
		int output_size = dnnLayoutGetMemorySize_F32(lt_user_input);
		fprintf(stderr, "MKLDNN_ConvertLayoutBackToNCHW init: N=%d,C=%d,H=%d,W=%d, mkldnnLayout = 0x%x, input_size = %d, output_size = %d\n", N,inC,inH,inW,mkldnnLayout,input_size,output_size);
#endif
		if(!dnnLayoutCompare_F32(mkldnnLayout, lt_user_input))
		{
			CHECK_ERR( dnnConversionCreate_F32(&cv_BacktoNCHW, mkldnnLayout, lt_user_input), err );
			CHECK_ERR( dnnAllocateBuffer_F32((void**)(&dnnbuffer), lt_user_input), err );
			primitives->storage->data[i*3] = (long long)cv_BacktoNCHW;
			primitives->storage->data[i*3 + 1] = (long long)dnnbuffer;
			primitives->storage->data[i*3 + 2] = (long long)(THTensor_(data)(input));
			torchbuffer = THTensor_(data)(input);
		}
	}
	else
	{
		cv_BacktoNCHW = (dnnPrimitive_t)primitives->storage->data[i*3];
		dnnbuffer = (real *)primitives->storage->data[i*3 + 1];
		torchbuffer = (real *)primitives->storage->data[i*3 + 2];
	}


	bool need_convert = (cv_BacktoNCHW != 0);
	real * buffer = NULL;
	if(need_convert)
	{
		//do not release the original buffer
		real * inPtr = THTensor_(data)(input);
		if(inPtr == torchbuffer)
		{
			//convert the torchbuffer to dnnbuffer
			buffer = dnnbuffer;
		}
		else
		{
			//convert the dnnbuffer to torchbuffer
			buffer = torchbuffer;
		}
		CHECK_ERR( dnnConversionExecute_F32(cv_BacktoNCHW, inPtr, buffer), err );
		input->storage->data = buffer;
		THStorage_(setMKLDNN)(input->storage);
	}
#if LOG_ENABLE
	fprintf(stderr, "MKLDNN_ConvertLayoutBackToNCHW:primitives = 0x%x, cv_BacktoNCHW = 0x%x, buffer = 0x%x, dnnbuffer = 0x%x, torchbuffer = 0x%x, need_convert = %d\n", primitives, cv_BacktoNCHW, buffer,dnnbuffer,torchbuffer,need_convert );
	fprintf(stderr, "MKLDNN_ConvertLayoutBackToNCHW: end. \n");
#endif

}

static void THNN_(SpatialConvolutionMM_MKLDNN_init_forward)(
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
          int outW,
          int group)

{
	dnnError_t err;
#if LOG_ENABLE
	fprintf(stderr, "SpatialConvolutionMM_MKLDNN_init_forward: start.");
	fprintf(stderr, "N=%d,inC=%d,inH=%d,inW=%d,kH=%d,kW=%d,dH=%d,dW=%d,padH=%d,padW=%d,outC=%d,outH=%d,outW=%d\n", N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW);
#endif
	dnnPrimitive_t m_conv_forward = NULL;
	dnnPrimitive_t m_conv_bwd_data = NULL;
	dnnPrimitive_t m_conv_bwd_filter = NULL;
	dnnPrimitive_t m_conv_bwd_bias = NULL;

	int f_dimension = dimension + (group != 1);
	size_t inputSize[dimension] = 	{inW,inH,inC,N};
	size_t filterSize[5] = 	{kW,kH,inC/group,outC/group,group};
	size_t outputSize[dimension] = 	{outW,outH,outC,N};
	size_t stride[dimension-2] = 	{dW,dH};
	int pad[dimension-2] = 		{-padW,-padH};

	size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };
	size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
	size_t filterStrides[5] = { 1, kW, kH * kW, (inC/group) * kH * kW, (inC/group)*(outC/group) * kH * kW };

	size_t biasSize[1] = { outputSize[2] };
	size_t biasStrides[1] = { 1 };

	//user layouts
	dnnLayout_t lt_user_input, lt_user_filter, lt_user_bias, lt_user_output;
	//forward layouts
	dnnLayout_t lt_forward_conv_input, lt_forward_conv_filter, lt_forward_conv_bias, lt_forward_conv_output;

	//forward conversions and buffers
	dnnPrimitive_t cv_forward_input = NULL,cv_forward_filter = NULL,cv_forward_bias = NULL,cv_forward_output = NULL;
	real * buffer_forward_input =NULL;real *buffer_forward_filter=NULL;real *buffer_forward_bias=NULL;real * buffer_forward_output =NULL;

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
#if CONVERSION_LOG
			fprintf(stderr ,"MKLDNN Convolution get input layout FAIL......\n");
#endif
		}
		else{
			lt_user_input = (dnnLayout_t)primitives->storage->data[CONV_LAYOUT_INPUT];
#if CONVERSION_LOG
			fprintf(stderr ,"MKLDNN Convolution get input layout OK\n");
#endif
		}
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_filter, f_dimension, filterSize, filterStrides), err );
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_bias, 1, biasSize, biasStrides) , err );
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides), err );

#if NEW_INTERFACE
		CHECK_ERR(dnnGroupsConvolutionCreateForwardBias_F32(&m_conv_forward, attributes, dnnAlgorithmConvolutionDirect, group, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
		CHECK_ERR(dnnGroupsConvolutionCreateBackwardData_F32(&m_conv_bwd_data, attributes, dnnAlgorithmConvolutionDirect, group, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
		CHECK_ERR(dnnGroupsConvolutionCreateBackwardFilter_F32(&m_conv_bwd_filter, attributes, dnnAlgorithmConvolutionDirect, group, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
		CHECK_ERR(dnnGroupsConvolutionCreateBackwardBias_F32(&m_conv_bwd_bias, attributes, dnnAlgorithmConvolutionDirect, group, dimension, outputSize),err);
#endif

		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_input, m_conv_forward, dnnResourceSrc) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_filter, m_conv_forward, dnnResourceFilter), err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_bias, m_conv_forward, dnnResourceBias) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_output,m_conv_forward, dnnResourceDst) , err );	

		//init forward conversions:
		CHECK_ERR( THNN_(init_conversion)(&cv_forward_input, 	&buffer_forward_input, 	lt_forward_conv_input, 	lt_user_input) , err );
		CHECK_ERR( THNN_(init_conversion)(&cv_forward_filter, 	&buffer_forward_filter, lt_forward_conv_filter, lt_user_filter), err );
		CHECK_ERR( THNN_(init_conversion)(&cv_forward_bias, 	&buffer_forward_bias, 	lt_forward_conv_bias, 	lt_user_bias), err );


		int size1 = dnnLayoutGetMemorySize_F32(lt_forward_conv_output);
		int size2 = dnnLayoutGetMemorySize_F32(lt_user_output);
		if(size1 == size2 && size2 == (outW*outH*outC*N*4))
		{
#if CONVERSION_LOG
			fprintf(stderr ,"MKLDNN Convolution forward ouput layout match OK\n");
#endif
		}
		else
		{
			if(!dnnLayoutCompare_F32(lt_forward_conv_output, lt_user_output))
			{
				CHECK_ERR( dnnConversionCreate_F32(&cv_forward_output, 	lt_forward_conv_output, lt_user_output), err );
			}
			CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forward_output), lt_forward_conv_output), err );
			fprintf(stderr, "MKLDNN Convolution forward output layout match FAIL: size1 = %d, size2 = %d, NCHW = %d \n",size1,size2,outW*outH*outC*N);
		}
		//primitives->storage->data[CONV_LAYOUT_INPUT] = lt_forward_conv_input;
	}
	else if(sizeof(real) == sizeof(double))
	{
#if NEW_INTERFACE
		CHECK_ERR(dnnConvolutionCreateForward_F64(&m_conv_forward,attributes, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
#else
		CHECK_ERR(dnnConvolutionCreateForward_F64(&m_conv_forward, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
#endif
	}

	//save the dnnPrimitive to THTensor(long int array)
	//save the output layout to dnnPrimitive
	primitives->storage->data[CONV_LAYOUT_FORWARD_OUTPUT] = (long long)lt_forward_conv_output;
	primitives->storage->data[CONV_LAYOUT_INPUT] = (long long)lt_forward_conv_input;
	primitives->storage->data[FORWARD_INDEX] = (long long)m_conv_forward;
	primitives->storage->data[BWD_DATA_INDEX] = (long long)m_conv_bwd_data;
	primitives->storage->data[BWD_FILTER_INDEX] = (long long)m_conv_bwd_filter;
	primitives->storage->data[BWD_BIAS_INDEX] = (long long)m_conv_bwd_bias;

	primitives->storage->data[CONVERT_FORWARD_INPUT] 	= (long long)cv_forward_input;
	primitives->storage->data[CONVERT_FORWARD_FILTER] 	= (long long)cv_forward_filter;
	primitives->storage->data[CONVERT_FORWARD_BIAS] 	= (long long)cv_forward_bias;
	primitives->storage->data[CONVERT_FORWARD_OUTPUT] 	= (long long)cv_forward_output;

	primitives->storage->data[BUFFER_FORWARD_INPUT] 	= (long long)buffer_forward_input;
	primitives->storage->data[BUFFER_FORWARD_FILTER] 	= (long long)buffer_forward_filter;
	primitives->storage->data[BUFFER_FORWARD_BIAS] 		= (long long)buffer_forward_bias;
	primitives->storage->data[BUFFER_FORWARD_OUTPUT] 	= (long long)buffer_forward_output;


#if LOG_ENABLE
	fprintf(stderr, "SpatialConvolutionMM_MKLDNN_init_forward: end, sizeof(real)=%d\n",sizeof(real));
#endif

}

static void THNN_(SpatialConvolutionMM_MKLDNN_init_bwddata)(
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
          int outW,int group)

{
	dnnError_t err;
#if LOG_ENABLE
	fprintf(stderr, "SpatialConvolutionMM_MKLDNN_init_bwddata: start.");
	fprintf(stderr, "N=%d,inC=%d,inH=%d,inW=%d,kH=%d,kW=%d,dH=%d,dW=%d,padH=%d,padW=%d,outC=%d,outH=%d,outW=%d\n", N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW);
#endif
	dnnPrimitive_t m_conv_bwd_data = NULL;

	int f_dimension = dimension + (group != 1);
	size_t inputSize[dimension] = 	{inW,inH,inC,N};
	size_t filterSize[5] = 	{kW,kH,inC/group,outC/group,group};
	size_t outputSize[dimension] = 	{outW,outH,outC,N};
	size_t stride[dimension-2] = 	{dW,dH};
	int pad[dimension-2] = 		{-padW,-padH};

	size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };
	size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
	size_t filterStrides[5] = { 1, kW, kH * kW, (inC/group) * kH * kW, (inC/group)*(outC/group) * kH * kW };
	size_t biasSize[1] = { outputSize[2] };
	size_t biasStrides[1] = { 1 };

	//user layouts
	dnnLayout_t lt_user_input, lt_user_filter, lt_user_bias, lt_user_output;
	//backward data layouts
	dnnLayout_t lt_bwddata_conv_input, lt_bwddata_conv_filter,lt_bwddata_conv_output;
	//backward data conversions and buffers
	dnnPrimitive_t cv_bwddata_input = NULL,cv_bwddata_filter = NULL,cv_bwddata_output = NULL;
	real * buffer_bwddata_input = NULL;real * buffer_bwddata_filter = NULL;real * buffer_bwddata_output=NULL;

#if NEW_INTERFACE
/*for new interface*/
	dnnPrimitiveAttributes_t attributes = NULL;
	CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );
#endif

	if(sizeof(real) == sizeof(float))
	{
		if(primitives->storage->data[CONV_LAYOUT_OUTPUT] == 0)
		{
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides), err );
#if CONVERSION_LOG
			fprintf(stderr ,"MKLDNN Convolution get output layout FAIL......\n");
#endif
		}
		else{
			lt_user_output = (dnnLayout_t)primitives->storage->data[CONV_LAYOUT_OUTPUT];
#if CONVERSION_LOG
			fprintf(stderr ,"MKLDNN Convolution get output layout OK\n");
#endif
		}
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_filter, f_dimension, filterSize, filterStrides), err );
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_bias, 1, biasSize, biasStrides) , err );
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides) , err );

		m_conv_bwd_data = (dnnPrimitive_t) (primitives->storage->data[BWD_DATA_INDEX]);
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwddata_conv_input, m_conv_bwd_data, dnnResourceDiffSrc) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwddata_conv_filter, m_conv_bwd_data, dnnResourceFilter) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwddata_conv_output, m_conv_bwd_data, dnnResourceDiffDst) , err );

		//get forward filter layout, convert from forward filter to bdwdata filter
		dnnPrimitive_t m_conv_forward = (dnnPrimitive_t)primitives->storage->data[FORWARD_INDEX];
		dnnLayout_t lt_forward_conv_filter = NULL;
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_filter, m_conv_forward, dnnResourceFilter), err );

		CHECK_ERR( THNN_(init_conversion)(&cv_bwddata_filter, 	&buffer_bwddata_filter, lt_bwddata_conv_filter, lt_forward_conv_filter) , err );
		CHECK_ERR( THNN_(init_conversion)(&cv_bwddata_output, 	&buffer_bwddata_output, lt_bwddata_conv_output, lt_user_output) , err );

                int size1 = dnnLayoutGetMemorySize_F32(lt_bwddata_conv_input);
                int size2 = dnnLayoutGetMemorySize_F32(lt_user_input);
                if(size1 == size2 && size2 == (inW*inH*inC*N*4))
                {
#if CONVERSION_LOG
                        fprintf(stderr ,"MKLDNN Convolution bwddata input layout match OK\n");
#endif
                }
                else
                {
			if(!dnnLayoutCompare_F32(lt_bwddata_conv_input, lt_user_input))
			{
				CHECK_ERR( dnnConversionCreate_F32(&cv_bwddata_input, lt_bwddata_conv_input, lt_user_input), err );
			}
                        CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_bwddata_input), lt_bwddata_conv_input), err );
                        //fprintf(stderr, "MKLDNN Convolution bwddata input layout match FAIL: size1 = %d, size2 = %d, NCHW = %d \n",size1,size2,inW*inH*inC*N*4);
                }

#if CONVERSION_LOG
		dnnLayout_t lt_conv_forward_output = (dnnLayout_t)primitives->storage->data[CONV_LAYOUT_FORWARD_OUTPUT];
		int check1 = dnnLayoutCompare_F32(lt_user_output, lt_bwddata_conv_output);
		int check2 = dnnLayoutCompare_F32(lt_user_output, lt_conv_forward_output);
		int check3 = dnnLayoutCompare_F32(lt_conv_forward_output, lt_bwddata_conv_output);
		int check4 = dnnLayoutCompare_F32(primitives->storage->data[CONV_LAYOUT_INPUT], lt_bwddata_conv_input);
		fprintf(stderr, "	MKLDNN Convolution backward data, check1=%d,check2=%d,check3=%d, check4=%d\n", check1,check2,check3,check4);
#endif
	}
	else if(sizeof(real) == sizeof(double))
	{
#if NEW_INTERFACE
		CHECK_ERR(dnnConvolutionCreateBackwardData_F64(&m_conv_bwd_data,attributes, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
#else
		CHECK_ERR(dnnConvolutionCreateBackwardData_F64(&m_conv_bwd_data, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
#endif
	}

	//save the dnnPrimitive to THTensor(long int array)
	//save the output layout to dnnPrimitive
	primitives->storage->data[CONV_LAYOUT_BWDDATA_INPUT] = (long long)lt_bwddata_conv_input;

	primitives->storage->data[CONVERT_BWDDATA_INPUT] 	= (long long)cv_bwddata_input;
	primitives->storage->data[CONVERT_BWDDATA_FILTER] 	= (long long)cv_bwddata_filter;
	primitives->storage->data[CONVERT_BWDDATA_OUTPUT] 	= (long long)cv_bwddata_output;

	primitives->storage->data[BUFFER_BWDDATA_INPUT] 	= (long long)buffer_bwddata_input;
	primitives->storage->data[BUFFER_BWDDATA_FILTER] 	= (long long)buffer_bwddata_filter;
	primitives->storage->data[BUFFER_BWDDATA_OUTPUT] 	= (long long)buffer_bwddata_output;

#if LOG_ENABLE
	fprintf(stderr, "SpatialConvolutionMM_MKLDNN_init_bwddata: end, sizeof(real)=%d\n",sizeof(real));
#endif

}

static void THNN_(SpatialConvolutionMM_MKLDNN_init_bwdfilter)(
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
          int outW,int group)

{
	dnnError_t err;
#if LOG_ENABLE
	fprintf(stderr, "SpatialConvolutionMM_MKLDNN_init_bwdfilter: start.");
	fprintf(stderr, "N=%d,inC=%d,inH=%d,inW=%d,kH=%d,kW=%d,dH=%d,dW=%d,padH=%d,padW=%d,outC=%d,outH=%d,outW=%d\n", N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW);
#endif
	dnnPrimitive_t m_conv_bwd_filter = NULL;

	int f_dimension = dimension + (group != 1);
	size_t inputSize[dimension] = 	{inW,inH,inC,N};
	size_t filterSize[5] = 	{kW,kH,inC/group,outC/group,group};
	size_t outputSize[dimension] = 	{outW,outH,outC,N};
	size_t stride[dimension-2] = 	{dW,dH};
	int pad[dimension-2] = 		{-padW,-padH};

	size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };
	size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
	size_t filterStrides[5] = { 1, kW, kH * kW, (inC/group) * kH * kW, (inC/group)*(outC/group) * kH * kW };

	size_t biasSize[1] = { outputSize[2] };
	size_t biasStrides[1] = { 1 };

	//user layouts
	dnnLayout_t lt_user_input, lt_user_filter, lt_user_bias, lt_user_output;
	//backward filter layouts
	dnnLayout_t lt_bwdfilter_conv_input, lt_bwdfilter_conv_filter,lt_bwdfilter_conv_output;
	//backward filter conversions and buffers
	dnnPrimitive_t cv_bwdfilter_input = NULL,cv_bwdfilter_filter = NULL,cv_bwdfilter_output = NULL;
	real * buffer_bwdfilter_input = NULL;real * buffer_bwdfilter_filter = NULL;real * buffer_bwdfilter_output = NULL;

#if NEW_INTERFACE
	dnnPrimitiveAttributes_t attributes = NULL;
	CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );
#endif

	if(sizeof(real) == sizeof(float))
	{

		//check the src and diffdst layout
		if(primitives->storage->data[CONV_LAYOUT_INPUT] == 0)
		{
#if CONVERSION_LOG
			fprintf(stderr ,"MKLDNN Convolution bwdfilter get input layout FAIL......\n");
#endif
			CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides) , err );
		}
		else
		{
#if CONVERSION_LOG
			fprintf(stderr ,"MKLDNN Convolution bwdfilter get input layout OK\n");
#endif
			lt_user_input = (dnnLayout_t)primitives->storage->data[CONV_LAYOUT_INPUT];
		}
		if(primitives->storage->data[CONV_LAYOUT_OUTPUT] == 0)
		{
#if CONVERSION_LOG
			fprintf(stderr ,"MKLDNN Convolution bwdfilter get output layout FAIL......\n");
#endif
			CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides), err );
		}
		else
		{

#if CONVERSION_LOG
			fprintf(stderr ,"MKLDNN Convolution bwdfilter get output layout OK\n");
#endif
			lt_user_output = (dnnLayout_t)primitives->storage->data[CONV_LAYOUT_OUTPUT];
		}
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_filter, f_dimension, filterSize, filterStrides), err );
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_bias, 1, biasSize, biasStrides) , err );

		m_conv_bwd_filter = (dnnPrimitive_t) (primitives->storage->data[BWD_FILTER_INDEX]);
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwdfilter_conv_input, m_conv_bwd_filter, dnnResourceSrc) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwdfilter_conv_filter, m_conv_bwd_filter, dnnResourceDiffFilter) , err );
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwdfilter_conv_output, m_conv_bwd_filter, dnnResourceDiffDst) , err );

		//init backward filter conversions:
		CHECK_ERR( THNN_(init_conversion)(&cv_bwdfilter_input, &buffer_bwdfilter_input, lt_bwdfilter_conv_input, lt_user_input) , err );
		CHECK_ERR( THNN_(init_conversion)(&cv_bwdfilter_output, &buffer_bwdfilter_output, lt_bwdfilter_conv_output, lt_user_output) , err );
		if(!dnnLayoutCompare_F32(lt_bwdfilter_conv_filter, lt_user_filter))
		{
			CHECK_ERR( dnnConversionCreate_F32(&cv_bwdfilter_filter, lt_bwdfilter_conv_filter, lt_user_filter), err );
			CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_bwdfilter_filter), lt_bwdfilter_conv_filter), err );			
		}

/*
		dnnLayout_t lt_conv_backward_input = (dnnLayout_t)primitives->storage->data[CONV_LAYOUT_BWDDATA_INPUT];
		int check1 = dnnLayoutCompare_F32(lt_user_input, lt_conv_backward_input);
		int check2 = dnnLayoutCompare_F32(lt_user_input, lt_bwdfilter_conv_input);
		int check3 = dnnLayoutCompare_F32(lt_bwdfilter_conv_input, lt_conv_backward_input);
		fprintf(stderr, "	MKLDNN Convolution backward filter, check1=%d,check2=%d,check3=%d\n", check1,check2,check3);
*/


	}
	else if(sizeof(real) == sizeof(double))
	{
#if NEW_INTERFACE
		CHECK_ERR(dnnConvolutionCreateBackwardFilter_F64(&m_conv_bwd_filter,attributes, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
#else
		CHECK_ERR(dnnConvolutionCreateBackwardFilter_F64(&m_conv_bwd_filter, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
#endif
	}

	//save the dnnPrimitive to THTensor(long int array)
	//save the output layout to dnnPrimitive
	primitives->storage->data[CONV_LAYOUT_BWDFILT_OUTPUT] = (long long)lt_bwdfilter_conv_filter;

	primitives->storage->data[CONVERT_BWDFILTER_INPUT] 	= (long long)cv_bwdfilter_input;
	primitives->storage->data[CONVERT_BWDFILTER_FILTER] 	= (long long)cv_bwdfilter_filter;
	primitives->storage->data[CONVERT_BWDFILTER_OUTPUT] 	= (long long)cv_bwdfilter_output;

	primitives->storage->data[BUFFER_BWDFILTER_INPUT] 	= (long long)buffer_bwdfilter_input;
	primitives->storage->data[BUFFER_BWDFILTER_FILTER] 	= (long long)buffer_bwdfilter_filter;
	primitives->storage->data[BUFFER_BWDFILTER_OUTPUT] 	= (long long)buffer_bwdfilter_output;



#if LOG_ENABLE
	fprintf(stderr, "SpatialConvolutionMM_MKLDNN_init_bwdfilter: end, sizeof(real)=%d\n",sizeof(real));
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
          int padH,
	  int group)
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
		primitives->storage->data[CONV_LAYOUT_INPUT] = (long long)input->mkldnnLayout;
		THNN_(SpatialConvolutionMM_MKLDNN_init_forward)(primitives,N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW,group);
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


	THTensor_(resize4d)(output, N, outC, outH, outW);
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
#if CONVERSION_LOG
			fprintf(stderr, "SpatialConvolutionMM_MKLDNN_forward conversion input \n");
#endif
			//optimize for input conversion, save the new layout , to avoid conversion in backward filter
		//	input->storage->data = buffer_forward_input;
		//	input->storageOffset = 0;
		//	input->mkldnnLayout = primitives->storage->data[CONV_LAYOUT_INPUT];
		}
		
		if(cv_forward_filter){
			resConv[dnnResourceFilter] = buffer_forward_filter;
			convert_resources[dnnResourceFrom] = filterPtr;
			convert_resources[dnnResourceTo]   = buffer_forward_filter;
			CHECK_ERR( dnnExecute_F32(cv_forward_filter, convert_resources), err );
#if CONVERSION_LOG
			fprintf(stderr, "SpatialConvolutionMM_MKLDNN_forward conversion filter \n");
#endif
		} 
		
		if(cv_forward_bias){
			resConv[dnnResourceBias] = buffer_forward_bias;
			convert_resources[dnnResourceFrom] = biasPtr;
			convert_resources[dnnResourceTo]   = buffer_forward_bias;
			CHECK_ERR( dnnExecute_F32(cv_forward_bias,convert_resources), err );
#if CONVERSION_LOG
			fprintf(stderr, "SpatialConvolutionMM_MKLDNN_forward conversion bias \n");
#endif
		} 
/*
		if(cv_forward_output){
			resConv[dnnResourceDst] = buffer_forward_output;
		} 
*/
		gettimeofday(&convert1,NULL);

		CHECK_ERR(dnnExecute_F32(m_conv_forward, (void**)resConv),err);	
		gettimeofday(&convert2,NULL);
/*
		if(cv_forward_output){
			//release the original buffer, replace it with the internal buffer
			if(output->mkldnnLayout == 0)
			{
				int memSize = output->storage->size;
				THStorage_(free)(output->storage);
				output->storage = THStorage_(newWithData)(buffer_forward_output,memSize);
			}
			output->storage->data = buffer_forward_output;
			output->storageOffset = 0;
		}
*/
		output->mkldnnLayout = (long long)primitives->storage->data[CONV_LAYOUT_FORWARD_OUTPUT];
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

#if MKL_TIME
	gettimeofday(&end,NULL);
	double duration1 = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	Convolution MKLDNN  forward time1 = %.2f ms \n",duration1);
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
	  int initOk,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,int group)
{
	struct timeval start,mid1,mid2,mid3,convert1,convert2,end;
	gettimeofday(&start,NULL);
	dnnError_t err;
	dnnPrimitive_t m_conv_bwdData =NULL; 

	dnnPrimitive_t cv_bwddata_input = NULL,cv_bwddata_filter = NULL,cv_bwddata_output = NULL;
	real * buffer_bwddata_input = NULL;real * buffer_bwddata_filter = NULL;real * buffer_bwddata_output=NULL;
	int N = input->size[0];
	int inC = input->size[1];
	int inH = input->size[2];
	int inW = input->size[3];

	int outC = gradOutput->size[1];
	int outH = gradOutput->size[2];
	int outW = gradOutput->size[3];
	if(initOk == 0)
	{
		primitives->storage->data[CONV_LAYOUT_OUTPUT] = (long long)gradOutput->mkldnnLayout;
		THNN_(SpatialConvolutionMM_MKLDNN_init_bwddata)(primitives,N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW,group);
	}


	//THTensor_(transpose)(weight, weight, 0, 1);

	gettimeofday(&mid1,NULL);

	m_conv_bwdData = (dnnPrimitive_t) (primitives->storage->data[BWD_DATA_INDEX]);
	cv_bwddata_input 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDDATA_INPUT];
	cv_bwddata_filter 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDDATA_FILTER];
	cv_bwddata_output  	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDDATA_OUTPUT];
	buffer_bwddata_input 	= (real *)(primitives->storage->data[BUFFER_BWDDATA_INPUT]);
	buffer_bwddata_filter 	= (real *)(primitives->storage->data[BUFFER_BWDDATA_FILTER]);
	buffer_bwddata_output 	= (real *)(primitives->storage->data[BUFFER_BWDDATA_OUTPUT]);

	real * buffer_forward_filter 	= (real *)(primitives->storage->data[BUFFER_FORWARD_FILTER]);

	THTensor_(resizeAs)(gradInput, input);
	gettimeofday(&mid2,NULL);
	//THTensor_(zero)(gradInput);



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
			//fprintf(stderr, "	MKLDNN Convolution backward data: outPtr=0x%x, buffer_bwddata_output=0x%x,cv_bwddata_output=0x%x \n",outPtr,buffer_bwddata_output,cv_bwddata_output );
			resConv[dnnResourceDiffDst] = buffer_bwddata_output;
			convert_resources[dnnResourceFrom] = outPtr;
			convert_resources[dnnResourceTo]   = buffer_bwddata_output;
			CHECK_ERR( dnnExecute_F32(cv_bwddata_output,convert_resources), err );
			//fprintf(stderr, "	MKLDNN Convolution backward data:	convert 1 done. \n");
		}


		if(cv_bwddata_filter){
			real * buffer_forward_filter 	= (real *)(primitives->storage->data[BUFFER_FORWARD_FILTER]);
			resConv[dnnResourceFilter] = buffer_bwddata_filter;
			convert_resources[dnnResourceFrom] = buffer_forward_filter;
			convert_resources[dnnResourceTo]   = buffer_bwddata_filter;
			CHECK_ERR( dnnExecute_F32(cv_bwddata_filter, convert_resources), err );
			//fprintf(stderr, "		convert 2 called. \n");
		}

		if(cv_bwddata_input){
			resConv[dnnResourceDiffSrc] = buffer_bwddata_input;
			gradInput->storageOffset = 0;
		}

		
		gettimeofday(&convert1,NULL);
		CHECK_ERR(dnnExecute_F32(m_conv_bwdData, (void**)resConv),err);	
		gettimeofday(&convert2,NULL);

		if(cv_bwddata_input){
			if(gradInput->mkldnnLayout == 0)
			{
				int memSize = gradInput->storage->size;
				THStorage_(free)(gradInput->storage);
				gradInput->storage = THStorage_(newWithData)(buffer_bwddata_input,memSize);
			}
			gradInput->storage->data = buffer_bwddata_input;
			THStorage_(setMKLDNN)(gradInput->storage);
		}

		gradInput->mkldnnLayout = (long long)primitives->storage->data[CONV_LAYOUT_BWDDATA_INPUT];

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
        fprintf(stderr,"        bwddata MKLDNN convert_time1 = %.2f ms, exec_time = %.2f, convert_time2=%.2fms \n",convert_time1,exec_time,convert_time2);


#endif

#if MKL_TIME
	gettimeofday(&end,NULL);
	double duration1 = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	Convolution MKLDNN  bwddata time1 = %.2f ms \n",duration1);
#endif

	//THTensor_(transpose)(weight, weight, 0, 1);
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
	  int initOk,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          real scale,int group)
{

	struct timeval start,mid,convert1,convert2,end;
	gettimeofday(&start,NULL);
	dnnError_t err;
	dnnPrimitive_t m_conv_bwdFilter =NULL, m_conv_bwdBias = NULL; 
	dnnPrimitive_t cv_bwdfilter_input = NULL,cv_bwdfilter_filter = NULL,cv_bwdfilter_output = NULL;
	real * buffer_bwdfilter_input = NULL;real * buffer_bwdfilter_filter = NULL;real * buffer_bwdfilter_output = NULL;

	int N = input->size[0];
	int inC = input->size[1];
	int inH = input->size[2];
	int inW = input->size[3];

	int outC = gradOutput->size[1];
	int outH = gradOutput->size[2];
	int outW = gradOutput->size[3];
	if(initOk == 0)
	{
		primitives->storage->data[CONV_LAYOUT_INPUT] = (long long)input->mkldnnLayout;
		primitives->storage->data[CONV_LAYOUT_OUTPUT] = (long long)gradOutput->mkldnnLayout;
		THNN_(SpatialConvolutionMM_MKLDNN_init_bwdfilter)(primitives,N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW,group);
	}

	m_conv_bwdFilter = (dnnPrimitive_t) (primitives->storage->data[BWD_FILTER_INDEX]);
	m_conv_bwdBias = (dnnPrimitive_t) (primitives->storage->data[BWD_BIAS_INDEX]);
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
	real * resBias[dnnResourceNumber]={0};
	resBias[dnnResourceDiffDst] = outPtr;
	resBias[dnnResourceDiffBias] = biasPtr;

	//fprintf(stderr, "	m_conv = 0x%x, inPtr=0x%x, filterPtr=0x%x, outPtr=0x%x\n", m_conv_bwdFilter,inPtr,filterPtr,outPtr);

	if(sizeof(real) == sizeof(float))
	{	
		//fprintf(stderr, "		cv1=0x%x, cv2=0x%x, cv3=0x%x\n",cv_user_to_conv1_input,cv_conv_to_user_filt,cv_user_to_conv1_output);
		//fprintf(stderr, "		buf1=0x%x, buf2=0x%x, buf3=0x%x\n",newInPtr,newFilterPtr,newOutPtr);

		if(cv_bwdfilter_input){
#if CONVERSION_LOG
			fprintf(stderr ,"MKLDNN Convolution bwdfilter input conversion\n");
#endif
			resConv[dnnResourceSrc] = buffer_bwdfilter_input;
			convert_resources[dnnResourceFrom] = inPtr;
			convert_resources[dnnResourceTo]   = buffer_bwdfilter_input;
			CHECK_ERR( dnnExecute_F32(cv_bwdfilter_input, convert_resources), err );
		}
		if(cv_bwdfilter_output){
#if CONVERSION_LOG
			fprintf(stderr ,"MKLDNN Convolution bwdfilter output conversion\n");
#endif
			resConv[dnnResourceDiffDst] = buffer_bwdfilter_output;
			convert_resources[dnnResourceFrom] = outPtr;
			convert_resources[dnnResourceTo]   = buffer_bwdfilter_output;
			CHECK_ERR( dnnExecute_F32(cv_bwdfilter_output, convert_resources), err );
		}
		if(cv_bwdfilter_filter){
			resConv[dnnResourceDiffFilter] = buffer_bwdfilter_filter;
		}


		gettimeofday(&convert1,NULL);
		CHECK_ERR(dnnExecute_F32(m_conv_bwdFilter, (void**)resConv),err);
		gettimeofday(&convert2,NULL);
	
		if(cv_bwdfilter_filter){
#if CONVERSION_LOG
			fprintf(stderr ,"MKLDNN Convolution bwdfilter filter conversion\n");
#endif
			convert_resources[dnnResourceFrom] = buffer_bwdfilter_filter;
			convert_resources[dnnResourceTo]   = filterPtr;
			CHECK_ERR( dnnExecute_F32(cv_bwdfilter_filter, convert_resources), err );
		}
		//fprintf(stderr, "		call float api:dnnExecute_F32 end, out[0]=%.2f \n",outPtr[0]);
		
		CHECK_ERR(dnnExecute_F32(m_conv_bwdBias, (void**)resBias),err);

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
        fprintf(stderr,"        bwdfilter MKLDNN convert_time1 = %.2f ms, exec_time = %.2f, convert_time2=%.2f ms\n",convert_time1,exec_time,convert_time2);

#endif

#if MKL_TIME
	gettimeofday(&end,NULL);
	double duration1 = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	Convolution MKLDNN  bwdfilter time1 = %.2f ms \n",duration1);
#endif

}

#endif
