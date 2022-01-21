Turing Create aims to combine the world of graphical editing with programming. Blender is a wonderful tool, but in it and other tools the GUI and Python worlds are somewhat separated.

The goal of Turing Create is to have a much tighter integration. GUI operations generate lines of code, vaguely similar to recording a macro, which can be then used and/or edited by any IDE.

So how is it different than macros? [Most macros]https://www.youtube.com/watch?v=zk7eOI33WCE&ab_channel=mathcodeprint pre-record a set of key/mouse inputs. [Writing a Blender addon]https://www.youtube.com/watch?v=hj3FtJsQFbA&ab_channel=Human_Robot3D is also useful, but the development of the addon itself is pure-programming (with an interactive "readout"). Secondly, neither of these tools can be integrated into a "world as code" paradigm. In Turing Create, individual actions will be much more context-aware and provide several choices as to which context we base off the input on. Finally, the code generated will be mutation-free using a shallow-copy-on-modify paradigm, enabling non-destructive workflows.

Another feature planned is custom sliders. You attach sliders to both to numerical literal values and optionally to in-world objects (much like the handles on shapes in Libreoffice) in order to manipulate them by dragging them in a scrollbar or in the 3D world, which can be easier than editing the number directly in some cases (you still can edit the numbers, of course).

**Working Submodules**

PythonPixels: This is the procedural texturing tool. It has the features of Blender's tools, and far more flexibility, but is much slower (for now). No GUI is available, but one is planned for the future.

**Partially-written Submodules**

TapeyTeapots: This is the mesh modeler tool. It is the main submodule and inspiration for Turing Create.

**Potential Future Submodules**

CurvesOfCode: A 2d vector tool inspired by Inkscape, but it may not be worth it to make this a separate tool from TapeyTeapots.

LatexLines: Word processing, probably using pyLatex. 

FunctionalFigures: Matplotlib wrapper.

ExternEngineer: Interface with other programs by calling API commands or even just sending out mouse and keyboard controls. The eventual goal is to add our tools to a GUI-only software.
