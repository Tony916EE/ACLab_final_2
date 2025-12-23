classdef sysID_UItest < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        DroneSysIDToolUIFigure        matlab.ui.Figure
        LaserSwitch                   matlab.ui.control.ToggleSwitch
        LaserSwitchLabel              matlab.ui.control.Label
        SystemIDDataGeneratorLabel    matlab.ui.control.Label
        TabGroup                      matlab.ui.container.TabGroup
        ConnectionTab                 matlab.ui.container.Tab
        Label                         matlab.ui.control.Label
        ResetDroneCameraButton        matlab.ui.control.Button
        ConnectDroneCameraButton      matlab.ui.control.Button
        CameraPreviewButton           matlab.ui.control.Button
        InitializationLabel           matlab.ui.control.Label
        SettingTab                    matlab.ui.container.Tab
        DistanceEditField             matlab.ui.control.NumericEditField
        DistanceEditFieldLabel        matlab.ui.control.Label
        RunButton                     matlab.ui.control.Button
        MinimumstepintervalDropDown   matlab.ui.control.DropDown
        MinimumstepintervalDropDownLabel  matlab.ui.control.Label
        InputamplitudeEditField       matlab.ui.control.NumericEditField
        InputamplitudeEditFieldLabel  matlab.ui.control.Label
        RuntimeEditField              matlab.ui.control.NumericEditField
        RuntimeEditFieldLabel         matlab.ui.control.Label
        ParameterSettingLabel         matlab.ui.control.Label
        UIAxes                        matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
        %% setting parameter initialization 
        time = 0; % time specify the total operation time 
        min_time = 0.25; % min_time specify the min_time interval hold in PBRS
        output = 0;  % output specify the amplitude of the input signal 
        direction = 1; % 1 represent the left/right direction  , 2 represent the up/down direction 
        %% drone Drinitialization 
        droneObj = 0;
        cameraObj = 0;
        initPos = [0 0 0];
        error = 0;
        distance = 450;
    end
    
    methods (Access = private)
        % 新增：偵測黃色便條紙的質心 (HSV 方法)
        function noteCentroid = detectNoteCentroid(app, frame)
            % 將 RGB 影像轉為 HSV
            hsvFrame = rgb2hsv(frame);
            % 設定黃色的色調範圍 (視需求調整: 此處示例值)
            hue = hsvFrame(:,:,1);
            sat = hsvFrame(:,:,2);
            val = hsvFrame(:,:,3);
            % 範例黃色範圍：H 在 [0.1,0.17] 之間 (約 36°-61°)
            maskYellow = hue > 0.1 & hue < 0.17 & sat > 0.5 & val > 0.5;
            % 取得連通區域性質
            props = regionprops(maskYellow, 'Centroid');
            if ~isempty(props)
                noteCentroid = props(1).Centroid;  % 取第一個物體中心
            else
                noteCentroid = [NaN, NaN];
            end
        end

        % 新增：偵測綠色雷射點的質心 (HSV 方法)
        function laserCentroid = detectLaserDot(app, frame)
            hsvFrame = rgb2hsv(frame);
            hue = hsvFrame(:,:,1);
            sat = hsvFrame(:,:,2);
            val = hsvFrame(:,:,3);
            % 範例綠色範圍：H 在 [0.25,0.4] 之間 (約 90°-144°)
            maskGreen = hue > 0.05 & hue < 0.45 & sat > 0.2 & val > 0.8;
            props = regionprops(maskGreen, 'Centroid');
            if ~isempty(props)
                laserCentroid = props(1).Centroid;
            else
                laserCentroid = [NaN, NaN];
            end
        end
        
        % 在 UIAxes 顯示 preview（取代彈出 figure）
        function showPreviewOnUIAxes(app, frame, centroidY, centroidG)
            try
                imshow(frame, 'Parent', app.UIAxes);
                hold(app.UIAxes, 'on');
                if ~any(isnan(centroidY))
                    plot(app.UIAxes, centroidY(1), centroidY(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
                end
                if ~any(isnan(centroidG))
                    plot(app.UIAxes, centroidG(1), centroidG(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
                end
                hold(app.UIAxes,'off');
            catch
                % 若 UIAxes 發生錯誤（例如未建立），就跳過
            end
        end

    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Value changed function: DistanceEditField
        function DistanceEditFieldValueChanged(app, event)
            app.distance = app.DistanceEditField.Value;
            
        end

        % Value changed function: RuntimeEditField
        function RuntimeEditFieldValueChanged(app, event)
            app.time = app.RuntimeEditField.Value;
            
        end

        % Value changed function: MinimumstepintervalDropDown
        function MinimumstepintervalDropDownValueChanged(app, event)
             %app.min_time = app.MinimumstepintervalDropDown.Value;
             app.min_time = 2;  %dt(0.2) * n
              
        end

        % Value changed function: InputamplitudeEditField
        function InputamplitudeEditFieldValueChanged(app, event)
            app.output = app.InputamplitudeEditField.Value;
            
        end

        % Callback function
        function LeftRightButtonValueChanged(app, event)

        end

        % Button pushed function: ResetDroneCameraButton
        function ResetDroneCameraButtonPushed(app, event)
    
             app.ConnectDroneCameraButton.Text = "Connect Drone & Camera";
             app.time = 0;
             app.output = 0;
             app.RuntimeEditField.Value = 0;
             app.InputamplitudeEditField.Value = 0;
             app.direction = 1;
             app.LeftRightButton.Text = "Left/Right";
             evalin("base",'clear drone');
             cla(app.UIAxes);
             
             
        end

        % Button pushed function: RunButton
        function RunButtonPushed(app, event)
            try
                disp("The program is running...")
                app.RunButton.Text = "Running...";
                % common params
                dt = 1/20; % for system ID
                % safe guard for zero time
                if app.time <= 0
                    app.time = app.RuntimeEditField.Value;
                end
        
                % check laser switch: assume Items = {'Off','On'}
                isLaserTracking = ischar(app.LaserSwitch.Value) && strcmp(app.LaserSwitch.Value, 'On');
        
                if isLaserTracking
                    % ======= Laser tracking mode =======
                    f = 820; Z = app.distance; dt_track = 0.3;
                    K = 14; S = 0.015; dead_px = 5;
                    prev_rc = [0 0 0 0];
                    failCount = 0; maxFail = 4;
                    startT = tic;
        
                    % try takeoff (if droneObj valid)
                    try
                        if isobject(app.droneObj)
                            takeoff(app.droneObj);
                            pause(1);
                            rc(app.droneObj,0,0,0,0);

                        end
                    catch
                        disp('Takeoff failed or drone not connected.');
                    end
                    disp("Takeoff end.")
                    while true
                        % stop conditions
                        if ~ischar(app.LaserSwitch.Value) || ~strcmp(app.LaserSwitch.Value, 'On')
                            disp('Laser tracking disabled by user');
                            break;
                        end
                        if toc(startT) > app.time
                            disp('Laser tracking timeout');
                            break;
                        end
        
                        % snapshot
                        try
                            frame = snapshot(app.cameraObj);
                        catch
                            frame = [];
                        end
                        if isempty(frame)
                            rc(app.droneObj,0,0,0,0);
                            pause(dt_track); continue;
                        end
        
                        % crop same ROI as calibrate/getPosition
                        if size(frame,2) >= 600
                            frame1 = frame(:,41:600,:);
                        elseif size(frame,2) > 40
                            frame1 = frame(:,41:end,:);
                        else
                            frame1 = frame;
                        end
        
                        centroidG = app.detectLaserDot(frame1); % [x_l,y_l]
                        centroidY = app.detectNoteCentroid(frame1); % [x_n,y_n]
        
                        % show preview
                        app.showPreviewOnUIAxes(frame1, centroidY, centroidG);
        
                        if any(isnan([centroidG, centroidY]))
                            rc(app.droneObj,0,0,0,0);
                            failCount = failCount + 1;
                            if failCount >= maxFail
                                try
                                    rc(app.droneObj,0,0,0,0);
                                    pause(0.2)
                                    land(app.droneObj);
                                catch
                                end
                                uialert(app.DroneSysIDToolUIFigure, '連續偵測失敗，已嘗試降落。','Error');
                                break;
                            end
                            pause(dt_track); continue;
                        else
                            failCount = 0;
                        end
        
                        x_l = centroidG(1); y_l = centroidG(2);
                        x_n = centroidY(1); y_n = centroidY(2);
        
                        ex = x_l - x_n; ey = y_l - y_n;
                        if abs(ex) < dead_px && abs(ey) < dead_px
                            rc(app.droneObj,0,0,0,0);
                            disp("laser catch");
                            pause(dt_track); continue;
                        end
        
                        dX = ex * Z / f;
                        dY = ey * Z / f;
                        vx = 1.3 * K * dX; vy = 2 * K * dY;
                        rc_roll  = S * vx;
                        rc_pitch = S * vy;
                        fprintf('rc_roll = %d\n', rc_roll);
                        fprintf('rc_pitch = %d\n', rc_pitch)
                        rc_cmd = 0.6*double(prev_rc) + 0.4*[rc_roll, 0, rc_pitch, 0];
                        rc_cmd = round(rc_cmd);          % 四捨五入
                        rc_cmd = min(max(rc_cmd, -100), 100); % 限制範圍
                        rc_cmd = int32(rc_cmd);          % 強制轉成整數型別
                        try
                            disp("Tracking begins");
                            rc(app.droneObj, -rc_cmd(1), 0, -rc_cmd(3), 0);
                            pause(dt_track);
                            disp(rc_cmd);
                        catch
                        end
                        prev_rc = rc_cmd;
                        % 顯示在 UIAxes
                        app.showPreviewOnUIAxes(frame1, centroidY, centroidG);
                    end
        
                    % exit tracking: stop and try land
                    try
                        disp("Tracking end.")
                        rc(app.droneObj,0,0,0,0); pause(0.5); land(app.droneObj);
                    catch
                    end
        
                else
                    rc(app.droneObj,0,0,0,0);
                    land(app.droneObj);
                    disp("The program is done.")
                end
        
                app.RunButton.Text = "Run";
            catch ME
                try
                    rc(app.droneObj,0,0,0,0); land(app.droneObj);
                catch
                end
                app.RunButton.Text = "Run";
                rethrow(ME);
            end
        end

        % Button pushed function: ConnectDroneCameraButton
        function ConnectDroneCameraButtonPushed(app, event)
            % clear the drone Obj in the workspace to avoid double 
            % connection error 
            app.ConnectDroneCameraButton.Text = "Connecting...";
            evalin("base","clear drone");
            pause(2)
            app.droneObj = 0;
            try 
                %% ==== drone initialization ==============================
                app.droneObj = ryze("Tello");
                app.cameraObj = webcam("Logitech StreamCam");
                assignin("base","drone",app.droneObj);
                %% ==== success and error handling ======================== 
                disp("the drone is connected !")
                app.ConnectDroneCameraButton.Text = "Connected!";
            catch ME
                % if an connection error is catch , the msgbox will pop out
                if strcmp(ME.identifier,'MATLAB:ryze:general:connectionFailed')
                    msgbox("Please check the wifi connection with Tello ","Error",'error')
                    app.ConnectDroneCameraButton.Text="Connect Drone & Camera";
                end 
            end 
        end

        % Button pushed function: CameraPreviewButton
        function CameraPreviewButtonPushed(app, event)
            try 
                frame = snapshot(app.cameraObj);
                % 取相同 ROI（若相機解析度小於 600 會自動處理）
                if size(frame,2) >= 600
                    frame1 = frame(:,41:600,:);
                elseif size(frame,2) > 40
                    frame1 = frame(:,41:end,:);
                else
                    frame1 = frame;
                end
        
                % 使用新函式做偵測
                centroidY = app.detectNoteCentroid(frame1);
                centroidG = app.detectLaserDot(frame1);
                % 顯示在 UIAxes
                app.showPreviewOnUIAxes(frame1, centroidY, centroidG);
        
            catch ME
                if strcmp(ME.identifier,"MATLAB:UndefinedFunction")
                    msgbox("Please execute connect first ","Error",'error')
                else
                    % 顯示錯誤訊息於命令列（方便 debug）
                    disp(ME.message)
                end 
            end 

        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create DroneSysIDToolUIFigure and hide until all components are created
            app.DroneSysIDToolUIFigure = uifigure('Visible', 'off');
            app.DroneSysIDToolUIFigure.AutoResizeChildren = 'off';
            app.DroneSysIDToolUIFigure.Position = [100 100 740 423];
            app.DroneSysIDToolUIFigure.Name = 'Drone SysID Tool';
            app.DroneSysIDToolUIFigure.Resize = 'off';

            % Create UIAxes
            app.UIAxes = uiaxes(app.DroneSysIDToolUIFigure);
            xlabel(app.UIAxes, 'time')
            ylabel(app.UIAxes, 'altitude')
            zlabel(app.UIAxes, 'Z')
            app.UIAxes.LabelFontSizeMultiplier = 1.2;
            app.UIAxes.FontWeight = 'bold';
            app.UIAxes.XTick = [];
            app.UIAxes.Position = [335 39 375 291];

            % Create TabGroup
            app.TabGroup = uitabgroup(app.DroneSysIDToolUIFigure);
            app.TabGroup.Position = [37 34 280 300];

            % Create ConnectionTab
            app.ConnectionTab = uitab(app.TabGroup);
            app.ConnectionTab.Title = 'Connection';
            app.ConnectionTab.BackgroundColor = [1 1 1];

            % Create InitializationLabel
            app.InitializationLabel = uilabel(app.ConnectionTab);
            app.InitializationLabel.FontSize = 16;
            app.InitializationLabel.FontWeight = 'bold';
            app.InitializationLabel.Position = [93 220 97 22];
            app.InitializationLabel.Text = 'Initialization';

            % Create CameraPreviewButton
            app.CameraPreviewButton = uibutton(app.ConnectionTab, 'push');
            app.CameraPreviewButton.ButtonPushedFcn = createCallbackFcn(app, @CameraPreviewButtonPushed, true);
            app.CameraPreviewButton.FontSize = 14;
            app.CameraPreviewButton.Position = [78 120 180 25];
            app.CameraPreviewButton.Text = 'Camera Preview';

            % Create ConnectDroneCameraButton
            app.ConnectDroneCameraButton = uibutton(app.ConnectionTab, 'push');
            app.ConnectDroneCameraButton.ButtonPushedFcn = createCallbackFcn(app, @ConnectDroneCameraButtonPushed, true);
            app.ConnectDroneCameraButton.FontSize = 14;
            app.ConnectDroneCameraButton.Position = [80 166 180 25];
            app.ConnectDroneCameraButton.Text = 'Connect Drone & Camera';

            % Create ResetDroneCameraButton
            app.ResetDroneCameraButton = uibutton(app.ConnectionTab, 'push');
            app.ResetDroneCameraButton.ButtonPushedFcn = createCallbackFcn(app, @ResetDroneCameraButtonPushed, true);
            app.ResetDroneCameraButton.FontSize = 14;
            app.ResetDroneCameraButton.Position = [72 76 180 25];
            app.ResetDroneCameraButton.Text = 'Reset Drone & Camera';

            % Create Label
            app.Label = uilabel(app.ConnectionTab);
            app.Label.HorizontalAlignment = 'center';
            app.Label.Position = [27 12 225 30];
            app.Label.Text = {'Please connect to the drone and camera '; 'before clicking connect button.'};

            % Create SettingTab
            app.SettingTab = uitab(app.TabGroup);
            app.SettingTab.Title = 'Setting';
            app.SettingTab.BackgroundColor = [1 1 1];

            % Create ParameterSettingLabel
            app.ParameterSettingLabel = uilabel(app.SettingTab);
            app.ParameterSettingLabel.FontSize = 16;
            app.ParameterSettingLabel.FontWeight = 'bold';
            app.ParameterSettingLabel.Position = [71 236 142 22];
            app.ParameterSettingLabel.Text = 'Parameter Setting';

            % Create RuntimeEditFieldLabel
            app.RuntimeEditFieldLabel = uilabel(app.SettingTab);
            app.RuntimeEditFieldLabel.HorizontalAlignment = 'right';
            app.RuntimeEditFieldLabel.Position = [84 164 53 22];
            app.RuntimeEditFieldLabel.Text = 'Run time ';

            % Create RuntimeEditField
            app.RuntimeEditField = uieditfield(app.SettingTab, 'numeric');
            app.RuntimeEditField.Limits = [0 Inf];
            app.RuntimeEditField.ValueChangedFcn = createCallbackFcn(app, @RuntimeEditFieldValueChanged, true);
            app.RuntimeEditField.Position = [152 164 100 22];

            % Create InputamplitudeEditFieldLabel
            app.InputamplitudeEditFieldLabel = uilabel(app.SettingTab);
            app.InputamplitudeEditFieldLabel.HorizontalAlignment = 'right';
            app.InputamplitudeEditFieldLabel.Position = [50 129 87 22];
            app.InputamplitudeEditFieldLabel.Text = 'Input amplitude';

            % Create InputamplitudeEditField
            app.InputamplitudeEditField = uieditfield(app.SettingTab, 'numeric');
            app.InputamplitudeEditField.Limits = [0 100];
            app.InputamplitudeEditField.RoundFractionalValues = 'on';
            app.InputamplitudeEditField.ValueChangedFcn = createCallbackFcn(app, @InputamplitudeEditFieldValueChanged, true);
            app.InputamplitudeEditField.Position = [152 129 100 22];

            % Create MinimumstepintervalDropDownLabel
            app.MinimumstepintervalDropDownLabel = uilabel(app.SettingTab);
            app.MinimumstepintervalDropDownLabel.HorizontalAlignment = 'right';
            app.MinimumstepintervalDropDownLabel.Position = [15 94 122 22];
            app.MinimumstepintervalDropDownLabel.Text = 'Minimum step interval';

            % Create MinimumstepintervalDropDown
            app.MinimumstepintervalDropDown = uidropdown(app.SettingTab);
            app.MinimumstepintervalDropDown.Items = {'0.05', '0.1', '0.2', '0.25', '0.5'};
            app.MinimumstepintervalDropDown.ValueChangedFcn = createCallbackFcn(app, @MinimumstepintervalDropDownValueChanged, true);
            app.MinimumstepintervalDropDown.Position = [152 94 100 22];
            app.MinimumstepintervalDropDown.Value = '0.25';

            % Create RunButton
            app.RunButton = uibutton(app.SettingTab, 'push');
            app.RunButton.ButtonPushedFcn = createCallbackFcn(app, @RunButtonPushed, true);
            app.RunButton.FontSize = 14;
            app.RunButton.Position = [92 17 100 25];
            app.RunButton.Text = 'Run';

            % Create DistanceEditFieldLabel
            app.DistanceEditFieldLabel = uilabel(app.SettingTab);
            app.DistanceEditFieldLabel.HorizontalAlignment = 'right';
            app.DistanceEditFieldLabel.Position = [80 199 57 22];
            app.DistanceEditFieldLabel.Text = 'Distance';

            % Create DistanceEditField
            app.DistanceEditField = uieditfield(app.SettingTab, 'numeric');
            app.DistanceEditField.Limits = [0 Inf];
            app.DistanceEditField.ValueChangedFcn = createCallbackFcn(app, @DistanceEditFieldValueChanged, true);
            app.DistanceEditField.Position = [152 199 100 22];
            app.DistanceEditField.Value = 450;

            % Create SystemIDDataGeneratorLabel
            app.SystemIDDataGeneratorLabel = uilabel(app.DroneSysIDToolUIFigure);
            app.SystemIDDataGeneratorLabel.FontSize = 24;
            app.SystemIDDataGeneratorLabel.FontWeight = 'bold';
            app.SystemIDDataGeneratorLabel.Position = [220 358 300 30];
            app.SystemIDDataGeneratorLabel.Text = 'System ID Data Generator';

            % Create LaserSwitchLabel
            app.LaserSwitchLabel = uilabel(app.DroneSysIDToolUIFigure);
            app.LaserSwitchLabel.HorizontalAlignment = 'center';
            app.LaserSwitchLabel.Position = [35 198 74 22];
            app.LaserSwitchLabel.Text = 'Laser Switch';

            % Create LaserSwitch
            app.LaserSwitch = uiswitch(app.DroneSysIDToolUIFigure, 'toggle');
            app.LaserSwitch.Position = [59 117 20 45];

            % Show the figure after all components are created
            app.DroneSysIDToolUIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = sysID_UItest

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.DroneSysIDToolUIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.DroneSysIDToolUIFigure)
        end
    end
end