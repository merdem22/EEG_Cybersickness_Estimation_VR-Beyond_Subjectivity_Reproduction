To make writing papers with me easy (i.e., more manageable for me), please follow this process.
There is no "I'm writing ML papers, so I don't have to follow it."
In stubborn cases, I reserve the right to postpone a submission to another deadline.
And please create your paper project as detailed below.


## our 2-stage process

Editing through whole drafts at the end of the process does not scale, it doesn't lead to the best results, and there's little to be learned on either side. My nights are also no longer available to turn drafts into papers.

### Stage 1: structure (i.e., the part that seems to be the most difficult)

* create the project as described below
* draft a *descriptive* title
  * focus on the problem and the contribution
  * don't be cute, Harry Potter and Lord of the Ring references were funny in undergrad

* draft an abstract
  * focus on the contribution, not the effort
  * it can include placeholders for results or datasets or ...

* define the Level 1 and Level 2(!) headings, and lower if suitable
  * particularly for the method and evaluation section
  * also useful to think about rough parts of discussion/conclusion at this point
  * also helpful to structure related work top-down
  * ➡️ this will require time and iteration!

* draft a Figure 1
  * again, could be a sketch or photo
  * Figure 1 takes more space and importance, so think about its composition
  * add a solid draft caption


* add placeholder figures and tables. they narrate the paper and plan for [final figures](paper-writing-figures). they can be
  * sketches
  * plots you have
  * plots you make up
  * photos (or quick placeholder or stock photos)
  * for tables, think about which columns to include
  * add captions!

* add [topic sentences](Paper-writing-in-LaTeX#topic-sentences) in the most important sections
  * introduction
  * method, especially the problem statement/definition
  * possibly evaluation if it requires more description
  * possibly discussion/implications
  * ➡️ these topic sentences can initially just be bullet points

➡️ share it for feedback. this is an excellent abstraction level to discuss the paper and will save all of us time, me having to review things, and you having to redo things.


### Stage 2: paper writing

* flesh out the full intro. share it for feedback
* fill the rest of the paper
* start creating [proper figures](paper-writing-figures)
* and follow [these rules](paper-writing-in-LaTeX)
* share it for feedback


## project organization

Keep the overall project clean.

* every section should be in a separate .tex file
* .tex files should be in a subfolder
* images etc. should be in one or more subfolders
* include our usual _defines.tex (and optionally _includes.tex), so we can use the same macros across papers

### organizing .tex files

* each sentence goes on a separate line
* two empty lines between paragraphs for better overview
* optional: add some empty lines (or % --------------------------------------------------) above a section, subsection, etc.
* if you want to keep stuff, but don't include it in the paper (anymore/yet), move it to the bottom of the file and prepend \endinput