# Define my Python user-defined exceptions


class PathDoesNotExistError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'The following path does not exist: {0} '.format(self.message)
        else:
            return 'Check the path as it does not exist.'


class EmptyDirectoryError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        print('calling str')
        if self.message:
            return 'There are no files at in this directory. {0} '.format(self.message)
        else:
            return 'There are no files at in this directory. Check setup.'


class NoLabelBoxProjectsFound(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'MyCustomError, {0} '.format(self.message)
        else:
            return 'MyCustomError has been raised'


class PropertiesDoesNotExistError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'The following properties file does not exist: {0} '.format(self.message)
        else:
            return 'Check the path as it does not exist.'

class UnhandledException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'Unhandled Exception: {0} '.format(self.message)
        else:
            return 'Unhandled Exception has been raised.'
