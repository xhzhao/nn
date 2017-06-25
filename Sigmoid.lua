local Sigmoid = torch.class('nn.Sigmoid', 'nn.Module')

function Sigmoid:updateOutput(input)
   local start=sys.clock()
   input.THNN.Sigmoid_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   print("Sigmoid_F = ", sys.clock() - start)
   return self.output
end

function Sigmoid:updateGradInput(input, gradOutput)
   local start=sys.clock()
   input.THNN.Sigmoid_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   print("Sigmoid_B = ", sys.clock() - start)
   return self.gradInput
end
