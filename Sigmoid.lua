local Sigmoid = torch.class('nn.Sigmoid', 'nn.Module')

function Sigmoid:updateOutput(input)
   local start=sys.clock()
   input.THNN.Sigmoid_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   sys.Sigmoid_F = sys.Sigmoid_F + sys.clock() - start

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
   sys.Sigmoid_B = sys.Sigmoid_B + sys.clock() - start
   return self.gradInput
end
