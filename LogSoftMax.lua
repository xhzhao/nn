local LogSoftMax = torch.class('nn.LogSoftMax', 'nn.Module')

function LogSoftMax:updateOutput(input)
   start=sys.clock()
   input.THNN.LogSoftMax_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   print("LogsoftMax_F = ", sys.clock() - start)
   return self.output
end

function LogSoftMax:updateGradInput(input, gradOutput)
   start=sys.clock()
   input.THNN.LogSoftMax_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   print("LogsoftMax_B = ", sys.clock() - start)
   return self.gradInput
end
