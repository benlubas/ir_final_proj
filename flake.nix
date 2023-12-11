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
            ]))

            nodePackages.pyright
          ];
        };
      }
    );
}
