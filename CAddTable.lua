local CAddTable, parent = torch.class('nn.CAddTable', 'nn.Module')

function CAddTable:__init(ip)
   parent.__init(self)
   self.inplace = ip
   self.gradInput = {}
end

function CAddTable:updateOutput(input)
   local start=sys.clock()
   if self.inplace then
      self.output:set(input[1])
   else
      self.output:resizeAs(input[1]):copy(input[1])
   end
   for i=2,#input do
      self.output:add(input[i])
   end
   self.t1 = self.t1 + sys.clock() - start
   self.count = self.count + 1
   if self.count == 100 then
      print("CAddTable_F = ", self.t1)
      print("CAddTable_B = ", self.t2)
      self.t1 = 0
      self.t2 = 0
      self.count = 0
   end


   return self.output
end

function CAddTable:updateGradInput(input, gradOutput)
   local start=sys.clock()
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[1].new()
      if self.inplace then
         self.gradInput[i]:set(gradOutput)
      else
         self.gradInput[i]:resizeAs(input[i]):copy(gradOutput)
      end
   end

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end
   self.t2 = self.t2 + sys.clock() - start
   return self.gradInput
end
