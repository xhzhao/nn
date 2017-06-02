#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Threshold.c"
#else

#define TH_OMP_THRESHOLD 10
void THNN_(Threshold_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real threshold,
          real val,
          bool inplace)
{
 
  if(input->nDimension >= 1)
  {
    long T = input->size[0];
    long t;

    if (inplace)
    {
#pragma omp parallel for if(T > TH_OMP_THRESHOLD) private(t)
      for(t = 0; t < T; t++)
      {
        THTensor *input_t = THTensor_(newSelect)(input, 0, t);  //slice and select

        TH_TENSOR_APPLY(real, input_t,
          if (*input_t_data <= threshold)
            *input_t_data = val;
        );
        THTensor_(free)(input_t);
        
      }
      THTensor_(set)(output, input);
    }
    else
    {
      THTensor_(resizeAs)(output, input);
#pragma omp parallel for if(T > TH_OMP_THRESHOLD) private(t)
      for(t = 0; t < T; t++)
      {
        THTensor *input_t = THTensor_(newSelect)(input, 0, t);  //slice and select
        THTensor *output_t = THTensor_(newSelect)(output, 0, t);
        TH_TENSOR_APPLY2(real, output_t, real, input_t,
          *output_t_data = (*input_t_data > threshold) ? *input_t_data : val;
        );

      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      }
    }
  }
  else
  {
     if (inplace)
    {
      TH_TENSOR_APPLY(real, input,
        if (*input_data <= threshold)
          *input_data = val;
      );
      THTensor_(set)(output, input);
    }
    else
    {
      THTensor_(resizeAs)(output, input);
      TH_TENSOR_APPLY2(real, output, real, input,
        *output_data = (*input_data > threshold) ? *input_data : val;
      );
    }

  }
}

void THNN_(Threshold_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real threshold,
          bool inplace)
{
   if(input->nDimension >= 2)
  {
  
    long T = input->size[0];
    long t;


    if (inplace)
    {
#pragma omp parallel for if(T > TH_OMP_THRESHOLD) private(t)
      for(t = 0; t < T; t++)
      {
        THTensor *input_t = THTensor_(newSelect)(input, 0, t);  //slice and select
        THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
        
        TH_TENSOR_APPLY2(real, gradOutput_t, real, input_t,
          if ((*input_t_data) <= threshold)
            *gradOutput_t_data = 0;
        );
        THTensor_(free)(input_t);
        THTensor_(free)(gradOutput_t);
        
      }
      THTensor_(set)(gradInput, gradOutput);
    }
    else
    {
      THTensor_(resizeAs)(gradInput, input);
#pragma omp parallel for if(T > TH_OMP_THRESHOLD) private(t)
      for(t = 0; t < T; t++)
      {
        THTensor *input_t = THTensor_(newSelect)(input, 0, t);  //slice and select
        THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
        THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
        TH_TENSOR_APPLY3(real, gradInput_t, real, gradOutput_t, real, input_t,
          if ((*input_t_data) > threshold)
            *gradInput_t_data = *gradOutput_t_data;
          else
            *gradInput_t_data = 0;
        );
        THTensor_(free)(input_t);
        THTensor_(free)(gradInput_t);
        THTensor_(free)(gradOutput_t);
      }
      
    }
  } 
  else
  {
    if (inplace)
    {
      TH_TENSOR_APPLY2(real, gradOutput, real, input,
        if ((*input_data) <= threshold)
          *gradOutput_data = 0;
      );
      THTensor_(set)(gradInput, gradOutput);
    }
    else
    {
      THTensor_(resizeAs)(gradInput, input);
      TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
        if ((*input_data) > threshold)
          *gradInput_data = *gradOutput_data;
        else
          *gradInput_data = 0;
      );
    }
  }
  
}

#endif
