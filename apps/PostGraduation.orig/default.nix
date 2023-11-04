{ pkgs ? import <nixpkgs> {} }:
with pkgs; stdenv.mkDerivation {
  name = "PostGraduation";
  propagatedBuildInputs =
    let pythondeps = with python38Packages;
          [
            djangorestframework
            django-cors-headers
            mysqlclient
          ];
    in [ python38 mysql ] ++ pythondeps;
  # shellHook = ./mysql.sh;
}
