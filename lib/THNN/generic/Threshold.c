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
  if (inplace)
  {
    if ( THTensor_(isContiguous)(input) )
    {
      real *tp = THTensor_(data)(input);
      long sz = THTensor_(nElement)(input);
      long j;
      real z;
      #pragma omp parallel for if(sz > TH_OMP_THRESHOLD) private(j)
      for (j=0; j<sz; j++)
      {
        if (tp[j] <= threshold)
          tp[j] = val;
      }
    }
    else
    {
      TH_TENSOR_APPLY(real, input,
        if (*input_data <= threshold)
          *input_data = val;
            );
    }
    THTensor_(set)(output, input);
  }
  else
  {
    THTensor_(resizeAs)(output, input);
    int dimI = input->nDimension;
    int dimO = output->nDimension;
    if ( (dimI==dimO) && THTensor_(isContiguous)(input) && THTensor_(isContiguous)(output) && THTensor_(nElement)(input) == THTensor_(nElement)(output))
    {
      real *tp = THTensor_(data)(input);
      real *rp = THTensor_(data)(output);
      long sz = THTensor_(nElement)(input);
      long j;
      real z;
      #pragma omp parallel for if(sz > TH_OMP_THRESHOLD) private(j) 
      for (j=0; j<sz; j++)
      {
        rp[j] = (tp[j] > threshold) ? tp[j] : val;
      }
    }
    else
    {
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
  if (inplace)
  {
    int dimI = input->nDimension;
    int dimO = gradOutput->nDimension;
    if ( (dimI==dimO) && THTensor_(isContiguous)(input) && THTensor_(isContiguous)(gradOutput) && THTensor_(nElement)(input) == THTensor_(nElement)(gradOutput))
    {
      real *tp = THTensor_(data)(input);
      real *rp = THTensor_(data)(gradOutput);
      long sz = THTensor_(nElement)(input);
      long j;
      real z;
      #pragma omp parallel for if(sz > TH_OMP_THRESHOLD) private(j) 
      for (j=0; j<sz; j++)
      {
        if (tp[j] <= threshold)
          rp[j] = 0;
      }
    }
    else
    {
      TH_TENSOR_APPLY2(real, gradOutput, real, input,
        if ((*input_data) <= threshold)
          *gradOutput_data = 0;
      );
    }

    THTensor_(set)(gradInput, gradOutput);
  }
  else
  {
    THTensor_(resizeAs)(gradInput, input);
    int dimI = input->nDimension;
    int dimO = gradOutput->nDimension;
    int dimG = gradInput->nDimension;
    if ( (dimI==dimO) && (dimI == dimG) && THTensor_(isContiguous)(input) && THTensor_(isContiguous)(gradOutput) && THTensor_(isContiguous)(gradInput) && THTensor_(nElement)(input) == THTensor_(nElement)(gradOutput) && THTensor_(nElement)(input) == THTensor_(nElement)(gradInput))
    {
      real *tp = THTensor_(data)(input);
      real *rp = THTensor_(data)(gradOutput);
      real *sp = THTensor_(data)(gradInput);
      long sz = THTensor_(nElement)(input);
      long j;
      real z;
      #pragma omp parallel for if(sz > TH_OMP_THRESHOLD) private(j)
      for(j=0; j < sz; j++)
      {
        if (tp[j] > threshold)
          sp[j] = rp[j];
        else
          sp[j] = 0;
      }
    }
    else
    {
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
