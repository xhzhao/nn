package = "nn"
version = "scm-1"

source = {
   url = "git://github.com/MlWoo/nn.git",
}

description = {
   summary = "Neural Network package for Torch",
   detailed = [[
   ]],
   homepage = "https://github.com/MlWoo/nn",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "luaffi",
   "moses >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DFORCE_AVX512=$(FORCE_AVX512)  -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"  -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_LIBDIR)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
