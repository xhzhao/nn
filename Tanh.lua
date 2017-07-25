local Tanh = torch.class('nn.Tanh', 'nn.Module')

function Tanh:updateOutput(input)
   local start=sys.clock()
   input.THNN.Tanh_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   sys.Tanh_F = sys.Tanh_F + sys.clock() - start

   return self.output
end

function Tanh:updateGradInput(input, gradOutput)
   local start=sys.clock()
   input.THNN.Tanh_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   sys.Tanh_B = sys.Tanh_B + sys.clock() - start
   return self.gradInput
end
