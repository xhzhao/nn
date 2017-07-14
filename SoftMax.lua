local SoftMax, _ = torch.class('nn.SoftMax', 'nn.Module')

function SoftMax:updateOutput(input)
   local start=sys.clock()
   input.THNN.SoftMax_updateOutput(
      input:cdata(),
      self.output:cdata()
   )

   self.t1 = self.t1 + sys.clock() - start
   self.count = self.count + 1
   if self.count == 500 then
      print("SoftMax_F = ", self.t1)
      print("SoftMax_B = ", self.t2)
      self.t1 = 0
      self.t2 = 0
      self.count = 0
   end

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
   self.t2 = self.t2 + sys.clock() - start
   return self.gradInput
end
