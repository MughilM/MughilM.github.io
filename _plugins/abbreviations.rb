# Site-wide Kramdown abbreviations.
#
# Kramdown only applies abbreviation definitions (`*[NLP]: ...`) within the file
# they are written in. This hook appends the shared list in
# _includes/abbreviations.md to every Markdown post and page before it is
# converted, so each abbreviation renders as <abbr title="..."> site-wide
# (styled in small caps by _sass/_benedikt.scss).
#
# A page can opt out with `abbreviations: false` in its front matter.
#
# Keep terms that appear inside $...$ maths OUT of the list: Kramdown would wrap
# them in <abbr> before KaTeX runs in the browser, which breaks the formula.

module SiteAbbreviations
  FILE = File.join("_includes", "abbreviations.md")

  def self.append(doc)
    return unless File.extname(doc.path.to_s) == ".md"
    return if doc.data["abbreviations"] == false

    path = File.join(doc.site.source, FILE)
    return unless File.exist?(path)

    doc.content = "#{doc.content}\n\n#{File.read(path)}"
  end
end

# :documents covers posts and every other collection (e.g. _projects).
Jekyll::Hooks.register [:pages, :documents], :pre_render do |doc|
  SiteAbbreviations.append(doc)
end
