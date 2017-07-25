local SoftMax, _ = torch.class('nn.SoftMax', 'nn.Module')

function SoftMax:updateOutput(input)
   local start=sys.clock()
   input.THNN.SoftMax_updateOutput(
      input:cdata(),
      self.output:cdata()
   )

   sys.SoftMax_F = sys.SoftMax_F + sys.clock() - start

   return self.output
end

function SoftMax:updateGradInput(input, gradOutput)
   local start=sys.clock()
   input.THNN.SoftMax_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   sys.SoftMax_B = sys.SoftMax_B + sys.clock() - start
   return self.gradInput
end
