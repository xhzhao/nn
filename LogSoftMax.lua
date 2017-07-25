local LogSoftMax = torch.class('nn.LogSoftMax', 'nn.Module')

function LogSoftMax:updateOutput(input)
   local start=sys.clock()
   input.THNN.LogSoftMax_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   sys.LogSoftMax_F = sys.LogSoftMax_F + sys.clock() - start

   return self.output
end

function LogSoftMax:updateGradInput(input, gradOutput)
   local start=sys.clock()
   input.THNN.LogSoftMax_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   sys.LogSoftMax_B = sys.LogSoftMax_B + sys.clock() - start
   return self.gradInput
end
