===============
📖 Bibliography
===============

DIPY uses the sphinxcontrib-bibtex_ Sphinx extension to manage the references
across the code base. This bibliography uses the bibtex_ format and is
gathered in a file named ``references.bib``.

The following conventions are used in that file.

- Entries are grouped according to the nature of the work: ``article`` types
  appear first, then e.g. ``dataset`` types, the ``electronic``, etc.
- Entries are sorted alphabetically within each group following the key of the
  entries.
- Keys are built using the last name of the first author and the year when the
  work was published, e.g. ``Garyfallidis2012``. If there are multiple works
  by the same author on the same year, a letter (e.g. ``a``, ``b``, etc.) is
  added after the year to distinguish them, e.g. ``Garyfallidis2012a``,
  ``Garyfallidis2012b``, starting with the least recent work (including works
  across groups). Within the same group, the least recent work is put closest
  to the end of the file. Similarly, if multiple works of the same author
  appeared in the same month, the name of the venue (i.e. its first letter)
  is used to distinguish works, following the rules described for the year.
- Strictly unnecessary (e.g. ``abstract``) or non-standard fields (``issn``)
  should be avoided.
- Author full names (i.e. including their first names) are used, and names use
  ``first name last name`` format.
- Authors are linked using the ``and`` (lowercase) conjunction.
- A blank line is added between two consecutive entries.
- Braces are used to hold the values of the keys.
- A whitespace line is added around the equal ``=`` operator in each field.
- The fields are aligned in a particular way for readability: the values of
  the fields are aligned.
- Entries should only contain ASCII characters.

Whenever this file needs to be edited (e.g. changed or a new entry added), the
above conventions must need to be followed.

.. include:: ../links_names.inc
