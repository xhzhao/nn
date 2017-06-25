#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Sigmoid.c"
#else

void THNN_(Sigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(sigmoid)(output, input);
}

void THNN_(Sigmoid_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THNN_CHECK_NELEMENT(output, gradOutput);
  THTensor_(resizeAs)(gradInput, output);
  ptrdiff_t gradInputSize = THTensor_(nElement)(gradInput);                     
  ptrdiff_t gradOutputSize = THTensor_(nElement)(gradOutput);   
  ptrdiff_t outputSize = THTensor_(nElement)(output);                    
  int gradInputContig = THTensor_(isContiguous)(gradInput)? 1:0;                 
  int gradOutputContig = THTensor_(isContiguous)(gradOutput)? 1:0;   
  int outputContig = THTensor_(isContiguous)(output)? 1:0; 
  if((gradInputSize = gradOutputSize) && (gradInputSize == outputSize) ){	
    TH_TENSOR_APPLY3_ADVANCED_INDEX2(outputSize, gradInputContig, gradOutputContig, outputContig,
                                     real, gradInput, real, gradOutput, real, output,
                                     real z = *output_data;
                                     *gradInput_data = *gradOutput_data * (1. - z) * z;
    );
  } else {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
      real z = *output_data;
      *gradInput_data = *gradOutput_data * (1. - z) * z;
    );
	  
  }
}

#endif
