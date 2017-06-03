local CAddTable, parent = torch.class('nn.CAddTable', 'nn.Module')

function CAddTable:__init(ip)
   parent.__init(self)
   self.inplace = ip
   self.gradInput = {}
end

function CAddTable:updateOutput(input)
   start=sys.clock()
   if self.inplace then
      self.output:set(input[1])
   else
      self.output:resizeAs(input[1]):copy(input[1])
   end
   for i=2,#input do
      self.output:add(input[i])
   end
   print("CAddTable_F = ", sys.clock() - start)
   return self.output
end

function CAddTable:updateGradInput(input, gradOutput)
   start=sys.clock()
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
   print("CAddTable_B = ", sys.clock() - start)
   return self.gradInput
end
