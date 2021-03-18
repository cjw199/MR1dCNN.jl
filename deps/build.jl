using Pkg
DIR = @__DIR__
@info "Installing packages..."
Pkg.activate(DIR*"/..")
Pkg.resolve()
Pkg.instantiate()
Pkg.status()

@info "Done building..."