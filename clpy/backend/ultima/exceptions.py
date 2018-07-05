class UltimaRuntimeError(RuntimeError):
    def __init__(self, status, stderr=''):
        self.status = status
        super(UltimaRuntimeError, self).__init__(
                'Return code:%d\nSTDERR:\n%s' % (status, stderr))

