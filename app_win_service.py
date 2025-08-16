import os
import sys
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
from api import app

class CarPricePredictionService(win32serviceutil.ServiceFramework):
    _svc_name_ = "CarPricePredictionService"
    _svc_display_name_ = "Car Price Prediction Service"
    _svc_description_ = "Hosts the Car Price Prediction API"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.host_name = socket.gethostname()
        self.server_name = None
        self.host = '0.0.0.0'
        self.port = 5000

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        if self.server_name:
            self.server_name.shutdown()
        self.ReportServiceStatus(win32service.SERVICE_STOPPED)

    def SvcDoRun(self):
        self.ReportServiceStatus(win32service.SERVICE_START_PENDING)
        try:
            self.ReportServiceStatus(win32service.SERVICE_RUNNING)
            self.server_name = app.run(host=self.host, port=self.port, debug=False)
        except Exception as e:
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            servicemanager.LogErrorMsg(f"Service failed to start: {str(e)}")
            return

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(CarPricePredictionService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(CarPricePredictionService)
