Python has been installed as
  /opt/homebrew/opt/python@3.8/bin/python3

Unversioned symlinks `python`, `python-config`, `pip` etc. pointing to
`python3`, `python3-config`, `pip3` etc., respectively, have been installed into
  /opt/homebrew/opt/python@3.8/libexec/bin

You can install Python packages with
  /opt/homebrew/opt/python@3.8/bin/pip3 install <package>
They will install into the site-package directory
  /opt/homebrew/lib/python3.8/site-packages

See: https://docs.brew.sh/Homebrew-and-Python

python@3.8 is keg-only, which means it was not symlinked into /opt/homebrew,
because this is an alternate version of another formula.

If you need to have python@3.8 first in your PATH, run:
  echo 'export PATH="/opt/homebrew/opt/python@3.8/bin:$PATH"' >> ~/.zshrc

For compilers to find python@3.8 you may need to set:
  export LDFLAGS="-L/opt/homebrew/opt/python@3.8/lib"

==> Summary
🍺  /opt/homebrew/Cellar/python@3.8/3.8.12_1: 4,453 files, 73.9MB
==> Running `brew cleanup python@3.8`...
Disable this behaviour by setting HOMEBREW_NO_INSTALL_CLEANUP.
Hide these hints with HOMEBREW_NO_ENV_HINTS (see `man brew`).