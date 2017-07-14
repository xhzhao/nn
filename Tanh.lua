local Tanh = torch.class('nn.Tanh', 'nn.Module')

function Tanh:updateOutput(input)
   local start=sys.clock()
   input.THNN.Tanh_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   self.t1 = self.t1 + sys.clock() - start
   self.count = self.count + 1
   if self.count == 500 then
      print("Tanh_F = ", self.t1)
      print("Tanh_B = ", self.t2)
      self.t1 = 0
      self.t2 = 0
      self.count = 0
   end

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
   self.t2 = self.t2 + sys.clock() - start
   return self.gradInput
end
