{ pkgs ? import <nixpkgs> { overlays = [ (self: super: import ./../../../deps/overlay.nix self super) ]; }
}:
with pkgs;
with python3Packages;
buildPythonPackage rec {
  name = "zhihu-backend";
  src = ./.;
  propagatedBuildInputs =
    [ django
      djangorestframework
      django-cors-headers
      # certifi
      # chardet
      django-filter
      djangorestframework-jwt
      # idna
      # pyjwt
      # pytz
      requests
      # urllib3
      # django-extensions
    ];
}
