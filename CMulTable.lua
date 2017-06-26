
local CMulTable, parent = torch.class('nn.CMulTable', 'nn.Module')

function CMulTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CMulTable:updateOutput(input)
   local start=sys.clock()
   self.output:resizeAs(input[1]):copy(input[1])
   for i=2,#input do
      self.output:cmul(input[i])
   end
   self.t1 = self.t1 + sys.clock() - start
   self.count = self.count + 1
   if self.count == 50 then
      print("CMulTable_F = ", self.t1)
      print("CMulTable_B = ", self.t2)
      self.t1 = 0
      self.t2 = 0
      self.count = 0
   end

   return self.output
end

function CMulTable:updateGradInput_efficient(input, gradOutput)
   self.tout = self.tout or input[1].new()
   self.tout:resizeAs(self.output)
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[1].new()
      self.gradInput[i]:resizeAs(input[i]):copy(gradOutput)
      self.tout:copy(self.output):cdiv(input[i])
      self.gradInput[i]:cmul(self.tout)
   end

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end

function CMulTable:updateGradInput(input, gradOutput)
   local start=sys.clock()
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[1].new()
      self.gradInput[i]:resizeAs(input[i]):copy(gradOutput)
      for j=1,#input do
         if i~=j then
            self.gradInput[i]:cmul(input[j])
         end
      end
   end

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end
   self.t2 = self.t2 + sys.clock() - start

   return self.gradInput
end

function CMulTable:clearState()
   if self.tout then self.tout:set() end
   return parent.clearState(self)
end
