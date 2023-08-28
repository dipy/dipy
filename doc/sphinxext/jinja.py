from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives

from jinja2 import FileSystemLoader, Environment

class JinjaDirective(Directive):
    
  # As this directive will bring some content into rst file
  has_contnet = True

  # To specify number of optional arguments
  optional_arguments = 0

  # To specify named optional arguments
  option_spec = {
      "file": directives.path,
      "context": directives.unchanged
  }

  # Defining app variable to access
  app = None
  
  def run(self):
    # Creating a node for our content
    node = nodes.Element()
    node.document = self.state.document
    
    # Fetching config
    conf = self.app.config
    
    # Fetching the context from conf.py
    cxt = (conf.jinja_contexts[self.options.get("context")].copy()
            if self.options.get("context") else {})
    
    # Loading the environment for Jinja parsing and new options
    env = Environment(
        loader=FileSystemLoader(conf.jinja_base, followlinks=True),
        **conf.jinja_env_kwargs
    )
    env.filters.update(conf.jinja_filters)
    env.tests.update(conf.jinja_tests)
    env.globals.update(conf.jinja_globals)
    env.policies.update(conf.jinja_policies)

    template_filename = self.options.get("file")

    if template_filename:
        tpl = env.get_template(template_filename)
    else:
        content = '\n'.join(self.content)
        tpl = env.from_string(content)

    # Rendering the content
    new_content = tpl.render(**cxt)

    # Converting it back to raw html
    # We can apply external classes from _static css files.
    return [nodes.raw('', new_content, format='html')]

def setup(app):
    JinjaDirective.app = app
    app.add_directive('jinja', JinjaDirective)
    app.add_config_value('jinja_contexts', {}, 'env')
    app.add_config_value('jinja_base', app.srcdir, 'env')
    app.add_config_value('jinja_env_kwargs', {}, 'env')
    app.add_config_value('jinja_filters', {}, 'env')
    app.add_config_value('jinja_tests', {}, 'env')
    app.add_config_value('jinja_globals', {}, 'env')
    app.add_config_value('jinja_policies', {}, 'env')


