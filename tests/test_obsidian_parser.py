import pytest

from src.core.models.document import Link
from src.infrastructure.filesystem.obsidian_parser import (
    ParsedDocument,
    parse_frontmatter,
    parse_inline_tags,
    parse_obsidian_document,
    parse_wikilinks,
    _strip_code_blocks,
)


class TestParseFrontmatter:
    def test_valid_yaml(self):
        content = "---\ntitle: Hello\ntags:\n  - foo\n  - bar\n---\nBody here"
        fm, body = parse_frontmatter(content)
        assert fm["title"] == "Hello"
        assert fm["tags"] == ["foo", "bar"]
        assert body == "Body here"

    def test_no_frontmatter(self):
        content = "Just a regular note\nwith content"
        fm, body = parse_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_empty_frontmatter(self):
        content = "---\n\n---\nBody"
        fm, body = parse_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_frontmatter_not_at_start(self):
        content = "Some text\n---\ntitle: Hello\n---\nBody"
        fm, body = parse_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_frontmatter_with_aliases(self):
        content = "---\ntitle: Test\naliases:\n  - alias1\n  - alias2\n---\nBody"
        fm, body = parse_frontmatter(content)
        assert fm["aliases"] == ["alias1", "alias2"]

    def test_frontmatter_trailing_whitespace(self):
        content = "---  \ntitle: Hello\n---  \nBody"
        fm, body = parse_frontmatter(content)
        assert fm["title"] == "Hello"
        assert body == "Body"

class TestStripCodeBlocks:
    def test_fenced_code_block(self):
        text = "before\n```python\ncode = 1\n```\nafter"
        result = _strip_code_blocks(text)
        assert "code = 1" not in result
        assert "before" in result
        assert "after" in result

    def test_inline_code(self):
        text = "use `#not-a-tag` in code"
        result = _strip_code_blocks(text)
        assert "#not-a-tag" not in result
        assert "use" in result

    def test_no_code(self):
        text = "plain text with [[link]] and #tag"
        assert _strip_code_blocks(text) == text

class TestParseWikilinks:
    def test_basic_link(self):
        links = parse_wikilinks("See [[My Note]] for details")
        assert len(links) == 1
        assert links[0].target == "My Note"
        assert links[0].target_heading is None
        assert links[0].target_block_id is None
        assert links[0].is_embed is False

    def test_link_with_alias(self):
        links = parse_wikilinks("See [[My Note|display text]]")
        assert len(links) == 1
        assert links[0].target == "My Note"

    def test_link_with_heading(self):
        links = parse_wikilinks("See [[My Note#Section One]]")
        assert len(links) == 1
        assert links[0].target == "My Note"
        assert links[0].target_heading == "Section One"
        assert links[0].target_block_id is None

    def test_link_with_block_id(self):
        links = parse_wikilinks("See [[My Note#^abc123]]")
        assert len(links) == 1
        assert links[0].target == "My Note"
        assert links[0].target_heading is None
        assert links[0].target_block_id == "abc123"

    def test_embed_link(self):
        links = parse_wikilinks("![[Embedded Note]]")
        assert len(links) == 1
        assert links[0].target == "Embedded Note"
        assert links[0].is_embed is True

    def test_embed_with_heading(self):
        links = parse_wikilinks("![[Note#Heading]]")
        assert len(links) == 1
        assert links[0].is_embed is True
        assert links[0].target_heading == "Heading"

    def test_same_note_heading(self):
        links = parse_wikilinks("See [[#Local Heading]]")
        assert len(links) == 1
        assert links[0].target == ""
        assert links[0].target_heading == "Local Heading"

    def test_same_note_block_id(self):
        links = parse_wikilinks("Ref [[#^myblock]]")
        assert len(links) == 1
        assert links[0].target == ""
        assert links[0].target_block_id == "myblock"

    def test_multiple_links(self):
        text = "Link to [[A]], [[B#H]], and ![[C#^d]]"
        links = parse_wikilinks(text)
        assert len(links) == 3
        assert links[0].target == "A"
        assert links[1].target == "B"
        assert links[1].target_heading == "H"
        assert links[2].target == "C"
        assert links[2].target_block_id == "d"
        assert links[2].is_embed is True

    def test_no_links(self):
        assert parse_wikilinks("plain text") == []

    def test_link_with_heading_and_alias(self):
        links = parse_wikilinks("[[Note#Heading|display]]")
        assert len(links) == 1
        assert links[0].target == "Note"
        assert links[0].target_heading == "Heading"

class TestParseInlineTags:
    def test_simple_tag(self):
        tags = parse_inline_tags("This is #important")
        assert tags == ["important"]

    def test_nested_tag(self):
        tags = parse_inline_tags("Filed under #project/alpha")
        assert tags == ["project/alpha"]

    def test_multiple_tags(self):
        tags = parse_inline_tags("#foo some text #bar and #baz")
        assert tags == ["foo", "bar", "baz"]

    def test_deduplication(self):
        tags = parse_inline_tags("#dup and #dup again")
        assert tags == ["dup"]

    def test_heading_not_a_tag(self):
        tags = parse_inline_tags("# Heading\n## Sub Heading\nSome #real-tag")
        assert tags == ["real-tag"]

    def test_tag_at_start_of_line(self):
        tags = parse_inline_tags("#start\ntext")
        assert tags == ["start"]

    def test_no_tags(self):
        assert parse_inline_tags("no tags here") == []

    def test_tag_with_underscores_and_hyphens(self):
        tags = parse_inline_tags("#my_tag-2")
        assert tags == ["my_tag-2"]

    def test_numeric_fragment_ignored(self):
        """A bare '#123' should not match since the first char must be [a-zA-Z_]."""
        tags = parse_inline_tags("issue #123")
        assert tags == []

class TestParseObsidianNote:
    SAMPLE_NOTE = """\
---
title: My Research
tags:
  - research
  - ai
aliases:
  - research-note
---
# Introduction

This note links to [[Deep Learning]] and [[Transformers#Attention]].

It also references a block ![[Paper Notes#^key-finding]].

## Tags

Inline tags: #machine-learning #ai

Some code example:
```python
x = "#not-a-tag"
link = "[[not-a-link]]"
```

Conclusion with `#also-not-a-tag` inline.
"""

    def test_frontmatter_extracted(self):
        result = parse_obsidian_document(self.SAMPLE_NOTE)
        assert result.frontmatter["title"] == "My Research"
        assert result.frontmatter["aliases"] == ["research-note"]

    def test_body_has_no_frontmatter(self):
        result = parse_obsidian_document(self.SAMPLE_NOTE)
        assert "---" not in result.body.split("\n")[0]
        assert "# Introduction" in result.body

    def test_links_extracted(self):
        result = parse_obsidian_document(self.SAMPLE_NOTE)
        targets = [(l.target, l.target_heading, l.target_block_id, l.is_embed) for l in result.links]
        assert ("Deep Learning", None, None, False) in targets
        assert ("Transformers", "Attention", None, False) in targets
        assert ("Paper Notes", None, "key-finding", True) in targets

    def test_links_inside_code_ignored(self):
        result = parse_obsidian_document(self.SAMPLE_NOTE)
        targets = [l.target for l in result.links]
        assert "not-a-link" not in targets

    def test_tags_merged_and_deduped(self):
        result = parse_obsidian_document(self.SAMPLE_NOTE)
        assert "research" in result.tags
        assert "ai" in result.tags
        assert "machine-learning" in result.tags
        # "ai" appears in both frontmatter and inline — only once
        assert result.tags.count("ai") == 1

    def test_tags_inside_code_ignored(self):
        result = parse_obsidian_document(self.SAMPLE_NOTE)
        assert "not-a-tag" not in result.tags
        assert "also-not-a-tag" not in result.tags

    def test_frontmatter_tags_come_first(self):
        result = parse_obsidian_document(self.SAMPLE_NOTE)
        assert result.tags.index("research") < result.tags.index("machine-learning")

    def test_note_without_frontmatter(self):
        result = parse_obsidian_document("Just #tagged content with [[Link]]")
        assert result.frontmatter == {}
        assert result.tags == ["tagged"]
        assert len(result.links) == 1
        assert result.links[0].target == "Link"

    def test_frontmatter_single_string_tag(self):
        note = "---\ntags: single\n---\nBody #inline"
        result = parse_obsidian_document(note)
        assert "single" in result.tags
        assert "inline" in result.tags

    def test_frontmatter_tags_with_hash_prefix(self):
        """Some users write tags as '#tag' in frontmatter YAML."""
        note = "---\ntags:\n  - '#prefixed'\n---\nBody"
        result = parse_obsidian_document(note)
        assert "prefixed" in result.tags
