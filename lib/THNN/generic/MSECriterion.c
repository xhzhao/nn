#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MSECriterion.c"
#else

#define TH_OMP_THRESHOLD 10
void THNN_(MSECriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage)
{
  real sum = 0;
  int dimI = input->nDimension;
  int dimO = target->nDimension;

  if ( (dimI==dimO) && THTensor_(isContiguous)(input) && THTensor_(isContiguous)(target) && THTensor_(nElement)(input) == THTensor_(nElement)(target))
  {
    real *tp = THTensor_(data)(input);
    real *rp = THTensor_(data)(target);
    long sz = THTensor_(nElement)(input);
    long j;
    real z;
    #pragma omp parallel for if(sz > TH_OMP_THRESHOLD) private(j) reduction(+:sum)
    for (j=0; j<sz; j++)
    {
      z = (rp[j] - tp[j]);
      sum += z*z;
    }
  }
  else
  {
    TH_TENSOR_APPLY2(real, input, real, target,
    real z = (*input_data - *target_data);
    sum += z*z;
      );
  }

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(MSECriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage)
{
  real norm = (sizeAverage ? 2./((real)THTensor_(nElement)(input)) : 2.);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
    *gradInput_data = norm * (*input_data - *target_data);
  );
}

#endif
