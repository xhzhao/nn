/*
* fileName: MKLDNN.h
* file description:
* 	integrate Torch + MKLDNN to accelerate CNN
* created by zhao xiaohui, intel@shanghai, email xiaohui.zhao@intel.com
*/

//#include <mkl.h>
#include "/opt/intel/mkl/include/mkl.h"
#include <sys/time.h>
/**
	MKLDNN init functions:
	SpatialConvolutionMM_MKLDNN_init()

	MKLDNN op run function:
	SpatialConvolutionMM_MKLDNN_forward()
	SpatialConvolutionMM_MKLDNN_bwdData()
	SpatialConvolutionMM_MKLDNN_bwdFilter()

	Primitives ara created in the init function and saved in the tensor.
	dnnPrimitives->storage->data[0]: forward
	dnnPrimitives->storage->data[1]: bwdData
	dnnPrimitives->storage->data[2]: bwdFilter
	...
*/
#define LOG_ENABLE 		0
#define NEW_INTERFACE		1

#define dimension 		4
#define FORWARD_INDEX 		0
#define BWD_DATA_INDEX 		1
#define BWD_FILTER_INDEX 	2

#define CONVERT_INPUT_INDEX 		3
#define CONVERT_FILTER_INDEX 		4
#define CONVERT_BIAS_INDEX 		5
#define CONVERT_CONV_OUTPUT_INDEX 	6
#define CONVERT_OUTPUT_INDEX 		7
#define CONVERT_BWDDATA_INDEX 		8
#define CONVERT_BWDFILTER_INDEX 	9

#define BUF_CONVERT_INPUT_INDEX 	10
#define BUF_CONVERT_FILTER_INDEX 	11
#define BUF_CONVERT_BIAS_INDEX 		12
#define BUF_CONVERT_OUTPUT_INDEX 	13


#define POOLING_FORWARD			0
#define POOLING_BACKWARD		1
//#define POOLING_CONVERT_OUTPUT	2
//#define POOLING_BUF_CONVERT_OUTPUT	3
#define POOLING_BUF_WORKSPACE		3
#define POOLING_BUF_DIFFSRC		4

#define RELU_FORWARD			0
#define RELU_BACKWARD			1

/*compare source define:
Convolution:1(forward),2(bwd data),3(bwd filter)
MaxPooling:4(forward),5(backward)
ReLU:6(forward),7(backward)
*/

#define CHECK_ERR(f, err) do { \
    (err) = (f); \
    if ((err) != E_SUCCESS) { \
        fprintf(stderr,"[%s:%d] err (%d)\n", __FILE__, __LINE__, err); \
    } \
} while(0)	



