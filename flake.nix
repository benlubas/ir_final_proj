{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {nixpkgs, flake-utils, ...}:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python310;
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            (python.withPackages (ps: with ps; [
              numpy
              nltk
              pandas
              black
              matplotlib
              # gensim
              ipykernel
              (
                buildPythonPackage rec {
                  pname = "tantivy";
                  version = "0.21.0";

                  src = fetchFromGitHub {
                    owner = "quickwit-oss";
                    repo = "tantivy-py";
                    rev = version;
                    hash = "sha256-hta6kPa2d4K1QnElrHpkDfUfpTg/mmAxNmTR1DOiY3Y=";
                  };

                  cargoDeps = rustPlatform.fetchCargoTarball {
                    inherit src;
                    name = "${pname}-${version}";
                    hash = "sha256-q0zXW+H62NG6zz/IxAvxz/L+uTg067rBTStPfd3laJg=";
                  };

                  format = "pyproject";

                  nativeBuildInputs = with rustPlatform; [ cargoSetupHook maturinBuildHook ];
                }
              )
            ]))

            nodePackages.pyright
          ];
        };
      }
    );
}
