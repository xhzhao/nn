local Tanh = torch.class('nn.Tanh', 'nn.Module')

function Tanh:updateOutput(input)
   start=sys.clock()
   input.THNN.Tanh_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   print("Tanh_F = ", sys.clock() - start)
   return self.output
end

function Tanh:updateGradInput(input, gradOutput)
   start=sys.clock()
   input.THNN.Tanh_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   print("Tanh_B = ", sys.clock() - start)
   return self.gradInput
end
