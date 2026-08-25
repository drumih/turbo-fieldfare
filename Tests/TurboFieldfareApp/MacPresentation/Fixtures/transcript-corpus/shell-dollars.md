Set the search path before running the tool.

```bash
echo $HOME
echo $PATH
export TARGET="$HOME/$PATH"
test $x = $y && echo ok
```

In prose the same variables appear as $HOME and $PATH, and the joined form
`"$HOME/$PATH"` stays inside a backtick span. A shell comparison such as
$x=$y on one line must not become an equation.
