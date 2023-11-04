self: super: rec {
  python3 = super.python3.override {
    packageOverrides = self: super: {
      # django
      django = super.buildPythonPackage rec {
        pname = "django";
        version = "2.2.28-patched";
        src = ./django22;
        doCheck = false;
        propagatedBuildInputs = with super;
          [ pytz sqlparse ];
      };

      # djangorestframework
      djangorestframework = super.buildPythonPackage rec {
        pname = "djangorestframework";
        version = "3.13.1-patched";
        src = ./restframework;
        doCheck = false;
        propagatedBuildInputs = with self;
          [ django ];
      };
    };
  };
  python3Packages = python3.pkgs;
}
