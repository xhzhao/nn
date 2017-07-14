local SplitTable, parent = torch.class('nn.SplitTable', 'nn.Module')

function SplitTable:__init(dimension, nInputDims)
   parent.__init(self)
   self.dimension = dimension
   self.nInputDims = nInputDims
end

function SplitTable:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input:dim() + dimension + 1
   elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end

function SplitTable:updateOutput(input)
   local start=sys.clock()
   local dimension = self:_getPositiveDimension(input)
   local slices = input:size(dimension)

   local currentOutput= {}
   for i=1,slices do
      currentOutput[#currentOutput+1] = input:select(dimension,i)
   end
   self.output = currentOutput
   self.t1 = self.t1 + sys.clock() - start
   self.count = self.count + 1
   if self.count == 500 then
      print("SplitTable_F = ", self.t1)
      print("SplitTable_B = ", self.t2)
      self.t1 = 0
      self.t2 = 0
      self.count = 0
   end

   return self.output
end 

function SplitTable:updateGradInput(input, gradOutput)
   local start=sys.clock()
   local dimension = self:_getPositiveDimension(input)
   local slices = input:size(dimension)
   if self.gradInput then
      self.gradInput:resizeAs(input)

      for i=1,slices do 
         local currentGradInput = gradOutput[i];        
         self.gradInput:select(dimension,i):copy(currentGradInput)
      end
   end
   self.t2 = self.t2 + sys.clock() - start
   return self.gradInput
end
