local Sigmoid = torch.class('nn.Sigmoid', 'nn.Module')

function Sigmoid:updateOutput(input)
   local start=sys.clock()
   input.THNN.Sigmoid_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   self.t1 = self.t1 + sys.clock() - start
   self.count = self.count + 1
   if self.count == 50 then
      print("Sigmoid_F = ", self.t1)
      print("Sigmoid_B = ", self.t2)
      self.t1 = 0
      self.t2 = 0
      self.count = 0
   end
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
   self.t2 = self.t2 + sys.clock() - start
   return self.gradInput
end
